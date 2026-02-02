import streamlit as st
import trafilatura
from urllib.parse import urljoin, urlparse
from groq import Groq
import os
import pandas as pd
from lxml import html
from trafilatura import html2txt
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import re
from groq.types.chat import ChatCompletionUserMessageParam
from datetime import datetime, date

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================= CONFIGURAÇÃO PÁGINA =========================

st.set_page_config(
    page_title=" Analisador de Conformidade",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= VALIDAÇÃO DA CHAVE DO GROQ============ (checado)
if "GROQ_API_KEY" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Chave da API do Groq não encontrada. Configure em secrets ou variável de ambiente.")
        st.stop()
    st.session_state.GROQ_API_KEY = api_key

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# ========================= LISTA DE MODELOS DE IA DO GROQ========= (ok)

# alguns modelos que estão disponíves no Groq
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "openai/gpt-oss-120b"
]

# ======================= LISTA DE CAMINHOS IRRELEVANTES ============= (ok)

LISTA_1 = [
    '/login', '/cadastro', '/conta', '/privacidade',
    '/contato', '/sobre', '/equipe', '/assinatura',
    '/webmail', '/galeria', '/simbolos'
          ]

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
        st.info("Selecione uma data de referência para ativar a análise contextualizada no período eleitoral.")

st.markdown("**Compare conteúdo de notícias de sites institucionais com normas eleitorais**")

st.divider()
# ========================================================================================
#                            Configurações do Modelo de IA
# ========================================================================================
if "modeloIA" not in st.session_state:
    st.session_state.modeloIA = GROQ_MODELS[0]

st.markdown("### IA (LLM)")
with st.expander("🤖 **Configurações do Modelo de IA**", expanded=False):
    col_model1, col_model2 = st.columns(2)
    with col_model1:
        modeloIA = st.selectbox(
            "Selecione o Modelo de IA (Groq)",
            options=GROQ_MODELS,
            index=0
        )
        modeloIA = st.session_state.modeloIA
    with col_model2:
        max_links = st.slider("Número máximo de LINKS por URL", 1, 20, 5, help="Quantos links internos seguir por site")
    temperatura = st.slider("Temperatura (criatividade)", 0.0, 2.0, 0.1, 0.1, help="O valor 0.0 é determinístico")

st.divider()

# ========================================================================================
#                                Adição e Lista de Sites (ok)
# ========================================================================================

st.markdown("### Adição de Sites")

if "sites_df" not in st.session_state:
    st.session_state.sites_df = pd.DataFrame(columns=["URL", "Nome do Site"])

# ====================== ADIÇÃO DE NOVO SITE ======================
def extrair_subdominio_gov(url: str) -> str:

    parsed = urlparse(url.strip())
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

# ====================== LISTA EDITÁVEL DE SITES ======================
    st.markdown("##### Lista de Sites para Análise")

    if st.session_state.sites_df.empty:
        st.info("Nenhum site adicionado ainda. Use o campo acima para incluir.")
    else:
        # Data editor com validação de duplicatas
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


# ==========================================================================##
#  ---------------------- INÍCIO BASE LEGAL ---------------------------------

# """
# Esse trecho do código é dedicado ao carregamento da normatização aplicável.
# A estrutura separada visa dimininuir a latência e reduzir a quantidade de tokens
# utilizados.
# A base de dados é trabalhada no mesmo ambiente de análise visando estabelecer
# uma conexão com o prompt de análise de conformidade dos conteúdos dos sites
# """



@st.cache_data(ttl=3600)
def resumir_base_legal(base_legal: str, data_referencia: str, modeloIA: str) -> str:
    if not base_legal.strip():
        return "Nenhuma base legal fornecida."

    prompt_base_legal = f"""
    Você é um jurista especializado em Direito Eleitoral.

    Dada a base legal completa abaixo e considerando a data de referência do pleito \"\"\"{data_referencia}\"\"\",

    Gere uma análise ESTRUTURADO, Denso e Hierárquico contendo APENAS as vedações, proibições e condutas permitidas/restritas aos
    agentes públicos no período eleitoral.
    
    ** Considere rigorosamente as condutas vedadas em função das datas de 6 e 3 meses que antecedem o pleito, pois, 
    há vedações como propaganda institucional, uso de bens públicos, etc, que dependem da data_referencia.

    Estrutura obrigatória do resumo (use exatamente este formato markdown para facilitar parsing):
    - **Vedações principais** (liste com bullets numerados ou -)
    - **Período de incidência** (datas relativas à eleição)
    - **Exceções e condutas permitidas**
    - **Sanções típicas** (breve)

    A análise não deve prejudicar a compreensão do conteúdo legal, por isso, além de completo, deve ser
     o mais fiel possível ao texto original, mas eliminando redundâncias e linguagem prolixa.
     
    Deixe bem claras as vedações correspondentes aos prazos de 3 e 6 meses que antecedem o pleito (data_referencia).

    Base legal completa:
    \"\"\"{base_legal}\"\"\"

    Responda APENAS com o documento da análise estruturada, sem introdução nem conclusão.
    """
    messages = [ChatCompletionUserMessageParam(role="user", content=base_legal)]

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

if "conteudo_base_legal" not in st.session_state:
    st.session_state.conteudo_base_legal = ""

st.markdown("### **Base Legal**")
with st.expander("📋 Base Legal", expanded=False):
    st.markdown("Defina o texto de referência legal que será usado na análise de conformidade pelo LLM.")

    # Carregar múltiplos TXT como referência

    st.markdown("### Upload arquivos .txt")
    st.markdown("**Carregue até 2 arquivos .txt** com trechos da lei, resolução, portaria, cartilha etc.")

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
        if conteudo_base_legal_referencia or texto_manual.strip():
            st.session_state.conteudo_base_legal = conteudo_base_legal_referencia
            if texto_manual.strip():
                st.session_state.conteudo_base_legal += "\n\n" + texto_manual.strip()
            st.info("Texto de referência pronto.")

            if st.button("Gerar Resumo da Base Legal"):
                with st.spinner("Resumindo base legal..."):
                    resumo = resumir_base_legal(
                        st.session_state.conteudo_base_legal,
                        st.session_state.data_referencia.strftime('%d/%m/%Y') if st.session_state.data_referencia else "não informada",
                        modeloIA
                    )
                    st.session_state.resumo_base_legal = resumo
                    st.success("Resumo gerado!")
                    st.markdown("**Resumo gerado:**")
                    st.markdown(resumo)

        # print(conteudo_base_legal_referencia)

#  ----------------- FIM PROCESSAMENTO BASE LEGAL ---------------------------
# ==========================================================================##


# ==========================================================================##
#  ---------------------- INÍCIO PROCESSAMENTO PROMPT ANÁLISE ----------------
# """


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
3 - Observe rigorosamente a data de início do pleito e as vedações correspondentes aos períodos de 3 e 6 meses que antecedem o pleito. As regras estão 
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
- Divida o texto em trechos significativos (frases ou parágrafos com ideia completa).
- Classifique cada trecho como "conforme" ou "não_conforme" com base no resumo.
- NÃO escreva NENHUM texto explicativo, introdução, conclusão, comentário ou palavra extra.
- Retorne EXATAMENTE cada trecho analisado para o processo de contagem, 
  sem aspas extras, sem JSON, sem formatação adicional.
_ Para cada trecho não conforme adicione à lista trechos_nao_conformes

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

st.markdown("### **Prompt**" )
with st.expander("🧠 Prompt", expanded=False):

    st.markdown("#### Prompt para Análise")

    if "prompt_reset" not in st.session_state:
        st.session_state.prompt_reset = 0

    prompt_personalizado = st.text_area(
        "Edite o prompt que será enviado ao modelo",
        value=prompt_padrao, # a variável prompt_personalizado recebe o conteúdo do prompt_padrao, alterado ou não
        height=350,
        key=f"prompt_editor_{st.session_state.prompt_reset}"
    )

#  ----------------- FIM PROCESSAMENTO PROMPT -------------------------------
# ==========================================================================##

st.divider()


# ============================================================
#  FUNÇÃO PARA COLETA DE LINKS DO SITE (ok)
# ============================================================

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

# ============================================================
#          FUNÇÃO PARA EXTRAÇÃO DE TEXTO     (ok)
# ============================================================

@st.cache_data(ttl=3600)
def extrair_texto(url_noticia: str) -> str:
    # 1. BAIXA HTML BRUTO — ESSENCIAL (ok)

    downloaded_noticia = trafilatura.fetch_url(url_noticia)

    if not downloaded_noticia:
        print(f"[ERRO] Falha ao baixar HTML bruto: {url_noticia}")
        return ""

    texto_final = None

    # 2. PRIMEIRA TENTATIVA — Trafilatura com máximo recall

    try:
        text_noticia = trafilatura.extract(
            downloaded_noticia,
            include_comments=False,
            include_images=False,
            include_tables=False,
            deduplicate=True,
            favor_recall=True,
            favor_precision=True,
            no_fallback=False,
            include_formatting=False,
            output_format="txt",
        )

        if text_noticia and len(text_noticia.strip()) > 150:
            texto_final = text_noticia # retorna uma str
        else:
            print(f"[WARN] Extração Trafilatura baixa em {url_noticia}")

    except Exception as e:
        print(f"[ERRO Trafilatura] {url_noticia}: {e}")

    # 3. FALLBACK 1 — html2txt (Trafilatura modo bruto)

    if not texto_final:
        try:
            print(f"[FALLBACK] html2txt ativado para {url_noticia}")
            raw_text = html2txt(downloaded_noticia)
            if raw_text and len(raw_text.strip()) > 150:
                texto_final = raw_text
        except:
            pass

    # 4. FALLBACK 2 — BeautifulSoup (captura TODO texto visível)

    if not texto_final:
        try:
            print(f"[FALLBACK] BeautifulSoup ativado para {url_noticia}")
            bs_noticia = BeautifulSoup(downloaded_noticia, "lxml")

            # Remove scripts, styles etc.
            for tag in bs_noticia(["script", "style", "noscript"]):
                tag.extract()

            bs_text = bs_noticia.get_text(separator="\n")
            if bs_text and len(bs_text.strip()) > 150:
                texto_final = bs_text

        except Exception as e:
            print(f"[ERRO BS4] {url_noticia}: {e}")

    if not texto_final:
        print(f"[ERRO] Nenhum método conseguiu extrair texto de {url_noticia}")
        return ""

    # print("texto extração")
    # print(texto_final)

    return texto_final

##### Esse trecho foi mantido para eventuais testes de raspagem feitos de forma mais crua.
# def extrair_texto(url: str) -> str:
#     # Baixa o HTML bruto
#     html = trafilatura.fetch_url(url)
#     if not html:
#         print(f"[ERRO] Não foi possível baixar: {url}")
#         return ""
#
#     # Extrai o texto no modo padrão
#     texto_final = trafilatura.extract(html)
#
#     return texto_final if texto_final else ""


# ==============================================================
#            FUNÇÃO PARA FILTRAR CONTEÚDO IRRELEVANTE
# ==============================================================

def filtrar_conteudo_relevante(texto: str) -> str:
    if not texto:
        return ""
    irrelevantes_keywords = [
        "política de privacidade", "cookies", "lgpd", "acessibilidade", "navegação", "teclas", "tab", "enter",
        "rolagem", "ctrl", "command", "razão social", "cnpj", "endereço", "contato", "login", "termos de uso",
        "sobre nós", "rodapé", "footer", "header", "menu", "navegador", "privacidade", "segurança", "captcha",
        "WhatsApp"
    ]
    # Remove seções inteiras que contenham palavras-chave
    blocos = re.split(r'\n\s*\n', texto)  # separa por parágrafos duplos
    blocos_filtrados = []
    for bloco in blocos:
        if not any(kw.lower() in bloco.lower() for kw in irrelevantes_keywords): # o que não está em bloco irrelevante passa.
            blocos_filtrados.append(bloco)
    return "\n\n".join(blocos_filtrados).strip()


# ============================================================
#  FUNÇÃO PARA ANÁLISE COM LLM - chamada da API do Groq (ok)
# ============================================================

def analisar_com_llm(texto: str,
                     model: str,
                     temperatura: float,
                     prompt_personalizado: str,
                     data_referencia):

    texto_filtrado = filtrar_conteudo_relevante(texto)
    if not texto_filtrado:
        return [], [0, 0, 0]

    data_ref_str = data_referencia.strftime('%d/%m/%Y') if data_referencia else "não informada"

    try:
        prompt_completo = prompt_personalizado.format(
            texto=texto_filtrado,
            data_referencia=data_ref_str,
            resumo_base_legal=st.session_state.get("resumo_base_legal", "Nenhum resumo disponível")
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
        print("=====================================content========================")
        print(content)

        trechos_nao_conformes = []
        contagem = [0, 0, 0]

        # Modificação 1: Expressão regular mais flexível
        match_trechos = re.search(r'trechos_nao_conformes\s*=\s*(\[.*?\])', content, re.DOTALL | re.IGNORECASE)
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
        match_contagem = re.search(r'contagem\s*=\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])', content, re.IGNORECASE)
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


# ========================================================================================
# --------------------------------ANÁLISE DOS SITES --------------------------------------
# ========================================================================================

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
                texto = extrair_texto(link)
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

# =====================================================================
# ========================= GRÁFICO DE BARRAS =========================
# =====================================================================

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
            ax.set_title(" 📊 Porcentagem de Conformidade", fontsize=10, pad=20)

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
