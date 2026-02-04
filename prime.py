import streamlit as st
import trafilatura
from urllib.parse import urljoin, urlparse
from groq import Groq
import os
import pandas as pd
from lxml import html
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import re
from groq.types.chat import ChatCompletionUserMessageParam
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
#        CONFIGURAÇÃO DA PÁGINA DO APLICATIVO
# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆


st.set_page_config(
    page_title=" Analisador de Conformidade",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
#        VALIDAÇÃO DA CHAVE DA API DO GROQ
# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆


if "GROQ_API_KEY" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Chave da API do Groq não encontrada. Configure em secrets ou variável de ambiente.")
        st.stop()
    st.session_state.GROQ_API_KEY = api_key

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# ◆━━━━━━━━━━━━━━  LISTA DE MODELOS DE IA ━━━━━━━━━━━━━━━━━━◆

# É possível incluir mais modelos que estão disponíveis no site

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "openai/gpt-oss-120b"
]

# ◆━━━━  CAMINHOS IRRELEVANTES PARA A BUSCA DE LINKS ━━━━━━━◆

LISTA_1 = [
    '/login', '/cadastro', '/conta', '/privacidade',
    '/contato', '/sobre', '/equipe', '/assinatura',
    '/webmail', '/galeria', '/simbolos'
          ]  # palavras-chave para exclusão na busca de links

# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
#                 CABEÇALHO DA PÁGINA
# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

col_titulo, col_data = st.columns(2)
with col_titulo:
    st.title("🗳️ Analisador de Conformidade Normativa")
with col_data:
    st.markdown("**Data de referência**")
    data_referencia = st.date_input(
        label="Período eleitoral de referência",
        value=None,  # sem valor padrão fixo → usuário deve escolher
        min_value=None,
        max_value=None,
        help="Selecione a data do primeiro turno).",
        format="DD/MM/YYYY"
    )
if data_referencia is not None:
    st.session_state.data_referencia = data_referencia
    st.caption(f"Data selecionada: **{data_referencia.strftime('%d/%m/%Y')}**")
else:
    st.session_state.data_referencia = None
    col_espaco, colAtivacaoDATA =st.columns(2)
    with colAtivacaoDATA:
        st.info("Selecione uma data de referência para ativar a análise contextualizada no período do defeso eleitoral.")

st.markdown("### Compare conteúdo de notícias de sites institucionais com normas eleitorais")

st.markdown("""
<hr style="border: 3px solid #666; margin: 20px 0;">
""", unsafe_allow_html=True)

# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆
#            SELEÇÃO E CONFIGURAÇÕES DA IA
# ◆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━◆

if "modeloIA" not in st.session_state:
    st.session_state.modeloIA = GROQ_MODELS[0]

st.markdown("### IA (LLM)")

with st.expander("🤖 **Configurações do Modelo de IA**", expanded=False):
    col_model1, col_model2 = st.columns(2)
    with col_model1:
        # seleciona o modelo de IA
        modeloIA = st.selectbox(
            "Selecione o Modelo de IA (Groq)",
            options=GROQ_MODELS,
            index=0
        )
        modeloIA = st.session_state.modeloIA
    with col_model2:
        # Define o máximo de links por URL que serão pesquisados
        max_links = st.slider("Número máximo de LINKS por URL", 1, 20, 5, help="Quantos links internos por site.")

    col_temp, col_caract = st.columns(2)
    with col_temp:
        # Define a temperatura para a LLM considerar a análise mais flexível (criativa) ou rígida (estatística)
        temperatura = st.slider("Temperatura (criatividade)", 0.0, 2.0, 0.1, 0.1, help="O valor 0.0 é determinístico.")
    with col_caract:
        # Define o número máximo de caracteres lidos para cada trecho da lido
        quant_caract = st.slider("Quantidade mínima de caracteres", 100, 500, 250, 50, help="Valores menores aumentam a quantidade de trechos para análise.")

# ◆━━━━━━━━━━━━   ADIÇÃO DE SITES   ━━━━━━━━━━━━━━━━━━━━━━━━◆

# Podem ser adicionado mais de um site

st.markdown("### Adição de Sites")

if "sites_df" not in st.session_state:
    st.session_state.sites_df = pd.DataFrame(columns=["URL", "Nome do Site"]) # monta a tabela com a lista das URLs

# ◆━━━━━━ EXTRAÇÃO DO SUBDOMÍNIO: MUN.UF.GOV.BR OU UF.GOV.BR ━━━━━━◆

def extrair_subdominio_gov(url: str) -> str:   # extrai o subdominio para facilitar a visualização

    parsed = urlparse(url.strip()) # limpa os espaços e desmonta a URL
    netloc = parsed.netloc.lower()

    if ':' in netloc:
        netloc = netloc.split(':')[0]
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    if not netloc.endswith('.gov.br'):
        raise ValueError(f"A URL não termina com .gov.br: {url}")
    dominio_sem_gov = netloc[:-7]
    partes = dominio_sem_gov.split('.')
    if len(partes) >= 2:
        resultado = '.'.join(partes[-2:])
    else:
        resultado = partes[-1]
    return resultado

# ◆━━━━━━━━━━━━━━━━━━━━━━━ ADIÇÃO DE NOVO SITE ━━━━━━━━━━━━━━━━━━━━━━━◆

with st.expander("🌐 sites", expanded=False):
    st.markdown("##### Adicionar novo site")
    col1, col2 = st.columns([3, 1])
    with col1:
        nova_url = st.text_input(
            "URL do site (ex: https://www.municipio.uf.gov.br/noticias)",
            placeholder="https://www.exemplo.go.gov.br/noticias -* https:// *- é mandatório",
            help="Página principal de notícias ou comunicados da administração pública."
        )

    if st.button("Adicionar Site", type="primary"):
        if not nova_url.strip():
            st.error("Por favor, insira uma URL válida.")
        else:
            url_limpa = nova_url.strip().rstrip("/")
            # Monta o dataframe com as URL/PATH
            urls_existentes = st.session_state.sites_df["URL"].str.rstrip("/").tolist()

            if url_limpa in urls_existentes:
                st.error("Esta URL já foi adicionada.")
            else:
                nome_exibicao = urlparse(url_limpa).netloc
                novo_site = pd.DataFrame([{
                    "URL": url_limpa,
                    "Nome do Site": nome_exibicao
                }])
                st.session_state.sites_df = pd.concat(
                    [st.session_state.sites_df, novo_site],
                    ignore_index=True
                )
                st.success(f"Site adicionado: {nome_exibicao}")
                st.rerun()

# ◆━━━━━━━━━━━━━━━━━━━━━━━ LISTA EDITÁVEL DE SITES ━━━━━━━━━━━━━━━━━━━━━━━◆

    st.markdown("##### Lista de Sites para Análise")

    if st.session_state.sites_df.empty:
        st.info("Nenhum site adicionado ainda. Use o campo acima para incluir.")
    else:
        # Aqui são apresentadas as URLs em uma tabela
        # data_editor com validação de duplicatas e com possibilidade de edição
        edited_df = st.data_editor(
            st.session_state.sites_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "URL": st.column_config.TextColumn(
                    "URL",
                    required=True,
                    help="URL completa da página de notícias"
                ),
                "Nome do Site": st.column_config.TextColumn(
                    "Nome do Site",
                    required=False,
                    help="Nome amigável para exibição"
                )
            },
            hide_index=True
        )

        # Validação: impedir URLs duplicadas ao editar
        urls_editadas = edited_df["URL"].str.strip().str.rstrip("/").tolist()
        if len(urls_editadas) != len(set(urls_editadas)):
            st.error("⚠️ Atenção: Não é permitido ter URLs duplicadas na lista.")
        else:
            # Só atualiza o estado se não houver duplicatas
            st.session_state.sites_df = edited_df
            st.success("Lista atualizada com sucesso!")
        # print(edited_df)
        st.caption(f"Total de sites: **{len(st.session_state.sites_df)}**")


# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ BASE LEGAL ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


#Esse trecho do código é dedicado ao carregamento da normatização aplicável.
#A estrutura separada visa dimininuir a latência e reduzir a quantidade de tokens
#utilizados.
#A base de dados é trabalhada no mesmo ambiente de análise dos sites visando estabelecer
#uma conexão com o prompt de análise de conformidade dos conteúdos dos sites.


@st.cache_data(ttl=3600) #decorator para carregar os dados na memória cache e evitar execuções repetidas
def analisar_base_legal(base_legal: str, data_referencia: str, modeloIA: str) -> str:
    if not base_legal.strip():
        return "Nenhuma base legal fornecida."

    prompt_base_legal = f"""
    Você é um jurista especializado em Direito Eleitoral.

    Dada a base legal completa de referência e considerando a data de referência do pleito \"\"\"{data_referencia}\"\"\",

    Gere uma análise ESTRUTURADA, Densa e Hierárquica destacando as vedações, proibições e condutas vedadas aos
    agentes públicos no período eleitoral, com base na data de referência informada pelo usuário.
    
    Calcule as datas para os períodos de defeso eleitoral que antecedem o primeiro pleito, considerando o primeiro período 6 meses que antecedem
    o pleito e o segundo período com 3 meses que antecedem o pleito. 
    
    ** Considere rigorosamente as condutas vedadas em função das datas que antecedem a data de referência por 6 e 3 meses, 
    analisando e identificando vedações como propaganda institucional, uso de bens públicos, etc, atreladas ao período de
    defeso eleitoral.

    Estrutura obrigatória da análise (use exatamente este formato markdown para facilitar parsing):
    - **Parágrafos com as Vedações principais** (liste com bullets numerados ou -)
    - **Indicações dos Períodos de incidência** (datas relativas à eleição)
    - **Parágrafos destacanto as Exceções e condutas permitidas**
    - **Parágrafos indicandos as Sanções típicas** (breve)

    A análise não deve prejudicar a compreensão do conteúdo legal, por isso, além de completo, deve ser
    o mais fiel possível ao texto da base legal carregada pelo usuário, SOMENTE ELIMINE redundâncias e linguagem prolixa.
     
    Deixe bem claras as vedações correspondentes aos prazos de 3 e 6 meses do defeso eleitoral que 
    antecedem a data do primeiro pleito (data_referencia).

    Base legal completa:
    \"\"\"{base_legal}\"\"\"

    Responda APENAS com o documento da análise estruturada, sem introdução nem conclusão.
    """
    # Carrega o prompt que será passado para análise pela LLM
    messages = [ChatCompletionUserMessageParam(role="user", content=prompt_base_legal)]

    #Parâmetros utilizados pela LLM via API
    try:
        response = client.chat.completions.create(
            model=modeloIA,
            messages=messages,
            temperature=0.1,  # baixa criatividade para fidelidade
            max_tokens=1000
        )
        print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Erro ao resumir base legal: {e}")
        return base_legal[:8000] + " [resumo truncado devido a erro]"

# inclui a variável conteudo_base_legal na seção do streamlit
if "conteudo_base_legal" not in st.session_state:
    st.session_state.conteudo_base_legal = ""

st.markdown("### **Base Legal**")
with st.expander("📋 Base Legal", expanded=False):
    st.markdown("Defina o texto de referência legal que será usado na análise de conformidade pelo LLM.")

    # Carregar múltiplos TXT como referência

    st.markdown("### Upload arquivos .txt")
    st.markdown("**Carregue até 2 arquivos .txt** com trechos da lei, resolução, portaria, cartilha etc.")

    # faz upload de arquivos do usuário em formato txt
    uploaded_txt_files = st.file_uploader(
        "Selecione arquivos TXT",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_referencia_multi",
        help="Máximo de 2 arquivos. Todos serão combinados em um único texto para a análise."
    )

    conteudo_base_legal_referencia = "" #declara como str

    if uploaded_txt_files:
        if len(uploaded_txt_files) > 2:
            st.error("Limite máximo: 2 arquivos TXT.")
            uploaded_txt_files = uploaded_txt_files[:2]

        textos_carregados = []
        for file in uploaded_txt_files:
            try:
                contenteudo_txt = file.read().decode("utf-8") # carrega o arquivo com conteúdo normativo .txt na variável
                # junta os conteúdo para formar a base legal
                textos_carregados.append(f"\n\n=== Conteúdo de: {file.name} ===\n{contenteudo_txt}") #lista de conteúdos
            except Exception as e:
                st.warning(f"Erro ao ler {file.name}: {e}")

        if textos_carregados:
            conteudo_base_legal_referencia = "\n".join(textos_carregados) #transfoma a lista textos_carregados em um só conteúdo
            st.success(f"{len(textos_carregados)} arquivo(s) TXT carregado(s) com sucesso.")
            st.caption(f"Total de caracteres: {len(conteudo_base_legal_referencia):,}")

        # Campo opcional para texto manual
        st.markdown("**Ou cole texto diretamente (opcional)**")
        texto_manual = st.text_area(
            "Texto adicional ou complementar.",
            height=150,
            placeholder="Cole aqui trechos específicos de julgados, artigos, doutrina etc."
        )

        # Texto final consolidado para a LLM
        # aqui a variável conteudo_base_legal recebe os valores de conteudo_base_legal_referencia ou texto_manual
        if conteudo_base_legal_referencia or texto_manual.strip():
            st.session_state.conteudo_base_legal = conteudo_base_legal_referencia
            if texto_manual.strip():
                st.session_state.conteudo_base_legal += "\n\n" + texto_manual.strip()
            st.info("Texto de referência pronto.")

            if st.button("Análise da Base Legal"):
                with st.spinner("Analisando a base legal..."):
                    analise_bl = analisar_base_legal(
                        st.session_state.conteudo_base_legal,
                        st.session_state.data_referencia.strftime('%d/%m/%Y') if st.session_state.data_referencia else "não informada",
                        modeloIA
                    )
                    st.session_state.analise_bl = analise_bl
                    st.success("Análise gerada!")
                    st.markdown("**Análise gerada:**")
                    st.markdown(analise_bl)



# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ FUNÇÕES AUXILIARES ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

# ◆━━━━━━━━━━━━━━━━━━━━━━━ FUNÇÃO PARA COLETA DE LINKS DO SITE ━━━━━━━━━━━━━━━━━━━━━━━◆

def coletar_links_internos(url: str, max_links) -> set:
    downloaded = trafilatura.fetch_url(url)  # web scraping
    if not downloaded:
        return {url}
    try:
        tree = html.fromstring(downloaded) # converte em uma árvore de dados hierárquicos
    except Exception:
        return {url}

    dominio = urlparse(url).netloc # extrai a parte da rede de uma URL
    links_validos = {url}

    #Loop para interar sobre todos os atributos href das tags de âncora (<a>) do tree.
    for href in tree.xpath("//a/@href"): #
        full = urljoin(url, href.strip())
        parsed = urlparse(full)

        if parsed.netloc != dominio: # Verifica se o domínio da URL extraída é o mesmo que o domínio da página original
            continue                 # se for diferente, ignora o link e não coleta o link externo.

        path = parsed.path.lower()

        if any(block in path for block in LISTA_1): # se verdadeiro ignora e não coleta o link
            continue

        if re.search(r'\.(pdf|jpg|jpeg|png|gif|zip|docx?|xlsx?)$', path): # se verdadeiro ignora e não coleta o link
            continue

        links_validos.add(full)

        if len(links_validos) >= max_links:
            break

    return links_validos

# ◆━━━━━━━━━━━━━━━━━━━━━━━ FUNÇÃO PARA EXTRAÇÃO DE TEXTO ━━━━━━━━━━━━━━━━━━━━━━━◆

@st.cache_resource(ttl=3600 * 4)  # Reutiliza browser → economia de recurso
def _get_playwright_browser():
    pw = sync_playwright().start()
    browser = pw.firefox.launch(headless=True, timeout=50000)
    return pw, browser

def extrair_texto(url: str, min_length) -> str:

    # Extração robusta para portais .gov.br:
    # Prioriza velocidade → fallback playwright só se necessário

    # Primeira tentativa -> leve e rápida
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return tentar_playwright(url, min_length)

    # A. Trafilatura otimizado (melhor recall em notícias)
    text = trafilatura.extract(
        downloaded,
        favor_recall=True,
        favor_precision=True,
        include_comments=False,
        include_tables=False,
        include_formatting=False,
        output_format="txt",
        no_fallback=False
    )
    if text and len(text.strip()) >= quant_caract:
        return limpar_texto(text)

    try:
        soup = BeautifulSoup(downloaded, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if len(text) >= min_length:
            return limpar_texto(text)
    except:
        pass

    # Último recurso: browser real (Playwright)
    return tentar_playwright(url, min_length)

def tentar_playwright(url: str, min_length: int) -> str:
    try:
        pw, browser = _get_playwright_browser()
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            locale="pt-BR",
            viewport={"width": 1280, "height": 900}
        )
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=35000)
        try:
            page.wait_for_load_state("networkidle", timeout=18000)
        except:
            pass

        # Rolagem leve para lazy-load
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        page.wait_for_timeout(800)

        # Tenta clicar em botões de aceite comuns
        for text in ["Aceitar", "Concordar", "OK", "Continuar", "Fechar", "Aceito"]:
            try:
                page.get_by_role("button", name=re.compile(text, re.I)).first.click(timeout=1800)
                break
            except:
                pass

        # Extrai via seleção de tags se conteúdo principais
        content = page.evaluate("""
            () => {
                const main = document.querySelector('article, main, [role="main"], #content, .entry-content, .post-content, .noticia-conteudo');
                return (main || document.body).innerText.trim();
            }
        """)

        page.close()
        context.close()

        if content and len(content) >= min_length:
            return limpar_texto(content)

    except (PWTimeoutError, Exception) as e:
        print(f"[PLAYWRIGHT falhou] {url} → {str(e)[:90]}")

    finally:
        # Não fecha o browser global aqui — reutilizado via cache_resource
        pass

    return ""


def limpar_texto(text: str) -> str:
    if not text:
        return ""
    # Remove blocos comuns que vazam em .gov.br
    text = re.sub(r'(?is)(política de (cookies|privacidade|lgpd)|acessibilidade|transparência ativa|ouvidoria|contato).*?(?=\n{2,}|$)', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

# ◆━━━━━━━━━━━━━━━━━━━━━━━ FUNÇÃO PARA FILTRAR CONTEÚDO IRRELEVANTE ━━━━━━━━━━━━━━━━━━━━━━━◆

# O objetivo do é filtrar os conteúdos que não correspondem a conteúdos estruturais da página

def filtrar_conteudo_relevante(texto: str) -> str:
    if not texto:
        return ""
    termos_irrelevantes = [
        "política de privacidade", "cookies", "lgpd", "acessibilidade", "navegação", "teclas", "tab", "enter",
        "rolagem", "ctrl", "command", "razão social", "cnpj", "endereço", "contato", "login", "termos de uso",
        "sobre nós", "rodapé", "footer", "header", "menu", "navegador", "privacidade", "segurança", "captcha",
        "WhatsApp"
    ]
    # Remove seções inteiras que contenham palavras-chave
    blocos = re.split(r'\n\s*\n', texto)  # separa por parágrafos duplos
    blocos_filtrados = []
    for bloco in blocos:
        if not any(k.lower() in bloco.lower() for k in termos_irrelevantes): # o que não está em bloco irrelevante passa.
            blocos_filtrados.append(bloco)
    return "\n\n".join(blocos_filtrados).strip()


# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ PROMPT PARA ANÁLISE DE CONTEÚDO DOS SITES ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

prompt_padrao = """
Você é um jurista especializado em compliance, com larga experiência em Direito Administrativo, Direito Eleitoral e 
ética na Administração Pública Federal.

Atue de forma técnica, objetiva, fundamentada e neutra, sem emitir juízos políticos ou valorativos.
[/PERSONA]

[CONTEXTO]
Durante o período eleitoral, é essencial que a Administração Pública observe rigorosamente as normas legais e éticas aplicáveis
às comunicações institucionais, bem como as condutas que são vedadas por lei, regulamento, norma etc. 

Para fins desta análise de conformidade, são considerados, EXCLUSIVAMENTE: 
1 - O texto passado pelo usuário por meio da variável "texto";
2 - a data do pleito passada por meio da variável "data_referencia"; e 
3 - O RESUMO PRÉVIO DA BASE LEGAL processado na etapa resumo da base legal.

[FLUXO]
Com base no texto, execute rigorosamente as seguintes etapas: 
1 - Divida o texto abaixo em trechos significativos (frases ou parágrafos com ideia completa e autônoma).
2 - Analise a conformidade de cada trecho com relação ao RESUMO PRÉVIO DA BASE LEGAL.
3 - Observe rigorosamente a data de início do pleito (data de referência informada pelo usuário) e as vedações correspondentes aos períodos de 3 e 6 meses que antecedem o pleito. As regras estão 
na resultado do processamento da base legal. 

RESUMO DA BASE LEGAL (referência única para julgar conformidade):
\"\"\"{resumo_base_legal}\"\"\"

INSTRUÇÕES RESTRIÇÃO SOBRE ELEMENTOS OU TAGs DE CONTEÚDOS EXTRAÍDOS – Desconsidere trechos cujo header traz uma dos seguintes termos:
- Ignore completamente links ou trechos que iniciem ou contenha de forma estrutural do html os seguintes termos: 
  'política de privacidade', 'cookies', 'LGPD', 'acessibilidade', 
  'navegação' '(TAB/ENTER/CTRL)', 'razão social', 'CNPJ', 'endereço', 'termos de uso', 'login'', 
  'contato', 'rodapé', 'menu', 'header',  'footer', "Acesse", "Serviços", "Órgão Vinculado", "Siga-nos" ou 
   qualquer elemento estrutural que não seja um texto com não-notícia.

- Foque apenas em notícias, comunicados ou textos institucionais relevantes.
- Divida o texto em trechos significativos (frases ou parágrafos com ideia completa e autônoma).
- Classifique cada trecho como "conforme" ou "não_conforme" com base no resumo. Seja muito rigoroso nessa parte, 
  os trechos com texto "conforme" é considerado para efeito do total de trechos. Ou seja, 
  o total de trechos deve obrigatoriamente sempre ser igual a soma dos trechos conformes e não conformes.
- Atenção na data de referencia informada pelo usuário, pois, a partir dela é que se considera os períodos do defeso eleitoral. 
  Não negligencie essa parte, é indispensável classificar a conformidade com relação aos períodos de defeso. 
  Exemplo: eventos, acontecimentos ou ações anteriores aos períodos de defeso informados na base legal podem ser desconsiderados. 
- NÃO escreva NENHUM texto explicativo, introdução, conclusão, comentário ou palavra extra.
- Retorne EXATAMENTE cada trecho analisado para o processo de contagem, 
  sem aspas extras, sem JSON, sem formatação adicional.
_ Para cada trecho não conforme adicione o trecho à lista trechos_nao_conformes.
- Se não houver nenhum techo não conforme, faça a variável total_conformes ter o valor igual a total_trechos_analisados

---------------------- RESULTADO ---------------------------------

A resposta final tem apenas 2 variáveis, trechos_nao_conformes e contagem, e deve-se seguir rigorosamente os seguintes formatos:

trechos_nao_conformes = [["trecho1 não conforme"], ["trecho2 não conforme"], ...]

contagem = [total_trechos_analisados, total_conformes, total_nao_conformes]

Exemplos obrigatórios do formato exato (copie exatamente):
Se houver 2 não conformes em 10 trechos (8 conformes):
trechos_nao_conformes = [["Texto do primeiro trecho não conforme"], ["Texto do segundo trecho não conforme"]]
contagem = [10, 8, 2]


Texto para análise:
\"\"\"{texto}\"\"\"

Data de referência:
\"\"\"{data_referencia}\"\"\"

Responda SOMENTE com as duas linhas acima. Nada mais.
"""

st.markdown("### **Prompt**")
with st.expander("🧠 Prompt", expanded=False):
    st.markdown("#### Prompt para Análise")

    if "prompt_reset" not in st.session_state:
        st.session_state.prompt_reset = 0

    prompt_personalizado = st.text_area(
        "Edite o prompt que será enviado ao modelo",
        # a variável prompt_personalizado recebe o conteúdo do prompt_padrao, que pode ser editado pelo usuário
        value=prompt_padrao,
        height=350,
        key=f"prompt_editor_{st.session_state.prompt_reset}"
    )


# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░ FUNÇÃO PARA ANÁLISE COM LLM - chamada da API do Groq ░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


def analisar_com_llm(texto: str,
                     model: str,
                     temperatura: float,
                     prompt_personalizado: str,
                     data_referencia):

    # extrai conteúdo relevante
    texto_filtrado = filtrar_conteudo_relevante(texto)
    if not texto_filtrado:
        return [], [0, 0, 0]

    data_ref_str = data_referencia.strftime('%d/%m/%Y') if data_referencia else "não informada"

    try:
        prompt_completo = prompt_personalizado.format(
            texto=texto_filtrado,
            data_referencia=data_ref_str,
            # resumo_base_legal=st.session_state.get("resumo_base_legal", "Nenhum resumo disponível")
            resumo_base_legal = st.session_state.get("resumo_base_legal")
        )
    except Exception as e:
        st.error(f"Erro no formato do prompt: {e}")
        return [], [0, 0, 0]

    try:
        messages = [ChatCompletionUserMessageParam(role="user", content=prompt_completo)]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperatura,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

        #print("=====================================content========================")
        #print(content)
        
        # Armazena os trechos não conformes e realiza a contagem global
        
        trechos_nao_conformes = []
        contagem = [0, 0, 0]

        # Modificação 1: Expressão regular mais flexível
        match_trechos = re.search(r'trechos_nao_conformes\s*=\s*(\[.*?])', content, re.DOTALL | re.IGNORECASE)
        if match_trechos:
            lista_str = match_trechos.group(1)
            # Limpar aspas e caracteres especiais
            lista_str = lista_str.replace('“', '"').replace('”', '"').replace("'", '"')
            # Remover quebras de linha dentro das strings
            lista_str = re.sub(r'\n', ' ', lista_str)
            try:
                lista_trechos = json.loads(lista_str)
                # Extrair strings das listas internas
                trechos_nao_conformes = []
                for item in lista_trechos:
                    if isinstance(item, list) and len(item) > 0:
                        trechos_nao_conformes.append(str(item[0]).strip())
                    elif isinstance(item, str):
                        trechos_nao_conformes.append(item.strip())
            except json.JSONDecodeError as e:
                print("Erro ao parsear trechos:", e, "\nConteúdo bruto:", lista_str)
                # Fallback: tentar extrair manualmente
                padrao_fallback = r'\[\s*"([^"]+)"\s*\]'
                trechos_encontrados = re.findall(padrao_fallback, lista_str)
                if trechos_encontrados:
                    trechos_nao_conformes = [t.strip() for t in trechos_encontrados]

        # Modificação 2: Expressão regular para contagem
        match_contagem = re.search(r'contagem\s*=\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*])', content, re.IGNORECASE)
        
        contagem = None
        contagem_str = None
        
        if match_contagem:
            try:
                contagem_str = match_contagem.group(1)
                contagem = json.loads(contagem_str)
            except:
                print("Erro ao parsear contagem:", match_contagem.group(1))
                # Fallback: extrair números
                numeros = re.findall(r'\d+', contagem_str)
                if len(numeros) >= 3:
                    contagem = [int(n) for n in numeros[:3]]

        return trechos_nao_conformes, contagem

    except Exception as e:
        st.warning(f"Erro na chamada ao LLM: {e}")
        return [], [0, 0, 0]



# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ANÁLISE DOS SITES ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


if "resultados" not in st.session_state:
    st.session_state.resultados = []

colAnalisar1, colAnalisar2, colAnalisar3 = st.columns([1, 2, 1])
with colAnalisar2:
    analisar = st.button("🚀 **Analisar Sites**", type="primary", use_container_width=True)

if analisar:
    if st.session_state.sites_df.empty:
        st.error("Adicione pelo menos um site antes de analisar.")
    else:
        sites = st.session_state.sites_df.to_dict("records")  # ok
        resultados_analise_llm = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        # print(sites)
        for idx, site in enumerate(sites):
            url = site["URL"]
            status_text.text(f"Analisando {idx + 1}/{len(sites)}: {url}")
            # print(max_links)
            links = coletar_links_internos(url, max_links=max_links)

            total_trechos = 0
            total_conformes = 0
            total_nao_conformes = 0
            trechos_nao_conformes = []

            # print(links)
            for link in links:
                texto = extrair_texto(link, quant_caract)
                if texto:
                    trechos_nao_conformes_site, lista_contagem = analisar_com_llm(
                        texto,
                        modeloIA,
                        temperatura,
                        prompt_personalizado,
                        st.session_state.data_referencia
                    )
                    # Acumula os trechos (lista de strings)
                    trechos_nao_conformes.extend(trechos_nao_conformes_site)

                    # Acumula contagens
                    total_trechos += lista_contagem[0]
                    total_conformes += lista_contagem[1]
                    total_nao_conformes += lista_contagem[2]

            # Calcula percentual de conformidade da URL
            if total_trechos == 0:
                perConformes = 0.0
            else:
                perConformes = round((total_conformes / total_trechos) * 100, 1)

            resultados_analise_llm.append({

                "url": url,
                "conformidade": perConformes,
                "total_trechos": total_trechos,
                "conformes": total_conformes,
                "nao_coformes": total_nao_conformes,
                "trechos_nao_conformes": trechos_nao_conformes

            })
            print("_____________________resultados_analise_llm___________________")
            print(resultados_analise_llm)

            progress_bar.progress((idx + 1) / len(sites))

        status_text.empty()
        progress_bar.empty()
        st.session_state.resultados = resultados_analise_llm


# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░░░░░░░ TABELA E GRÁFICO DE BARRAS DOS RESULTADOS ░░░░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░


resultados_para_plot = st.session_state.get("resultados", [])

if resultados_para_plot:
    def nome_grafico(url):
        return extrair_subdominio_gov(url)


    df_result = pd.DataFrame({
        "Site": [nome_grafico(r.get("url", "")) for r in resultados_para_plot],
        "Conformidade (%)": [float(r.get("conformidade", 0.0)) for r in resultados_para_plot]
        # chave correta é "conformidade"
    })

    df_result = df_result.dropna(subset=["Site", "Conformidade (%)"])

    trechos_nao_conformes = []

    for resultado in resultados_para_plot:
        url = resultado.get("url", "—")
        nome_site = nome_grafico(url)
        trechos = resultado.get("trechos_nao_conformes", [])  # lista de strings

        for trecho in trechos:
            if isinstance(trecho, str) and trecho.strip():
                trechos_nao_conformes.append({
                    "Site": nome_site,
                    "Trecho": trecho.strip(),
                    "Classificação": "nao_conforme",
                    "URL original": url
                })

    if trechos_nao_conformes:
        df_nao_conformes = pd.DataFrame(trechos_nao_conformes)
        df_nao_conformes = df_nao_conformes.drop_duplicates()

        st.divider()
        st.subheader("🟥 Trechos identificados como possível indício de conduta vedada")

        # Exibe a tabela interativa (com filtro, ordenação, etc.)
        st.dataframe(
            df_nao_conformes,
            column_config={
                "Site": st.column_config.TextColumn("Site", width="medium"),
                "Trecho": st.column_config.TextColumn("Trecho identificado", width="large"),
                "Classificação": st.column_config.TextColumn("Classif.", width="small"),
                "URL original": st.column_config.LinkColumn("URL", width="medium", display_text=r"https?://(.+)")
            },
            hide_index=True,
            use_container_width=True
        )

        # contador rápido trechos
        st.caption(f"Total de trechos com não conformidades: **{len(df_nao_conformes)}**")

        # Botão para baixar CSV
        csv = df_nao_conformes.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar tabela como CSV",
            data=csv,
            file_name="trechos_indicio.csv",
            mime="text/csv"
        )
    else:
        st.info("Nenhum trecho classificado como 'não conforme' foi encontrado na análise.")

    if not df_result.empty:
        col_esq, col_centro, col_dir = st.columns([1, 2, 1])

        with col_centro:
            fig, ax = plt.subplots(figsize=(10, 5))

            sites = df_result["Site"]
            valores = df_result["Conformidade (%)"].astype(float).clip(0, 100)

            # Cores por gradiente
            cores = plt.colormaps['viridis'](valores / 100.0)

            bars = ax.bar(sites, valores, color=cores, edgecolor='blue', linewidth=0.8)

            # Rótulos com percentual
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

            ax.set_xlabel("")
            ax.set_ylabel("Conformidade (%)", fontsize=10)
            ax.set_title(" 📊 Grau de Conformidade dos Trechos Analisados", fontsize=10, pad=20)

            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8)

            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylim(0, 100)

            plt.tight_layout()
            st.pyplot(fig)

# Rodapé
st.markdown("---")
st.caption("Analisador de Conformidade de Conduta Vedada | Desenvolvido por Fabiana, João Vicente, Lívia, Túlio e Yroá")



