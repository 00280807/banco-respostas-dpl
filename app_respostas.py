# app_respostas.py  (versão com Google Sheets backend)
import streamlit as st
import pandas as pd
import os
import json
import gspread
from gspread_dataframe import set_with_dataframe
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- LINHAS DE TESTE: mostrar chaves carregadas de st.secrets ---
st.write("SECRETS CARREGADOS:", list(st.secrets.keys()))
# --------------------------------------------------------------

# ---------------- Config visual (mantém seu tema) ----------------
st.set_page_config(page_title="Banco de Respostas da DPL", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    body { background-color: #F9F9F6; color: #333333; }
    .stApp { background-color: #F9F9F6; }
    header[data-testid="stHeader"] { background-color: #1B5E20; }
    [data-testid="stSidebar"] { background-color: #E8F5E9; }
    h1,h2,h3,h4 { color: #1B5E20; }
    .css-18e3th9 { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.image(
    "https://www.gov.br/icmbio/pt-br/assuntos/biodiversidade/unidade-de-conservacao/unidades-de-biomas/marinho/lista-de-ucs/parna-marinho-dos-abrolhos/fomulario-denuncia/icmbio-logo-1.png/@@images/93d85e33-e72b-423a-bc35-5d1b1f09b402.png",
    width=180
)
st.title("Banco de Respostas da DPL")
st.caption("🌿 Harmonizando manifestações institucionais com inovação e gestão do conhecimento")

# ---------------- Config do Google Sheets via Secrets ----------------
# Você deve ter adicionado em Streamlit Secrets:
# gcp_service_account = '''{ ... JSON da service account ... }'''
# sheet_url = "https://docs.google.com/spreadsheets/d/SEU_ID_AQUI/edit#gid=0"

if "gcp_service_account" not in st.secrets:
    st.error("Erro: credencial gcp_service_account não encontrada nos Secrets. Configure conforme instruções.")
    st.stop()
if "sheet_url" not in st.secrets:
    st.error("Erro: sheet_url não encontrada nos Secrets.")
    st.stop()

# Carregar credenciais da secret (JSON)
service_account_info = json.loads(st.secrets["gcp_service_account"])
# Autenticar gspread
gc = gspread.service_account_from_dict(service_account_info)
# Abrir planilha por URL
SHEET = gc.open_by_url(st.secrets["sheet_url"])
# Usaremos a primeira worksheet
ws = SHEET.sheet1

# ---------------- Modelo semântico ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")
model = load_model()

# ---------------- Funções que usam Google Sheets ----------------
COLS = [
    "Nº do processo SEI",
    "Tipo do documento",
    "Nº do documento",
    "Autoria",
    "Texto do documento recebido",
    "Texto da resposta institucional enviada"
]

def carregar_banco():
    # Lê todas as linhas da planilha e transforma em DataFrame
    try:
        records = ws.get_all_records()
        if len(records) == 0:
            return pd.DataFrame(columns=COLS)
        df = pd.DataFrame.from_records(records)
        # Garantir colunas certas (se faltar, adicionar)
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[COLS]
        return df
    except Exception as e:
        st.error(f"Erro ao carregar planilha: {e}")
        return pd.DataFrame(columns=COLS)

def salvar_banco(df):
    # Escreve o dataframe inteiro na planilha (substitui o conteúdo)
    try:
        # Se a planilha está vazia ou com cabeçalho, primeiro limpar
        ws.clear()
        # Escreve o dataframe com cabeçalho
        set_with_dataframe(ws, df, include_index=False, include_column_header=True)
    except Exception as e:
        st.error(f"Erro ao salvar na planilha: {e}")

# ---------------- Funções do app (adição, busca, visualização/edição) ----------------
def adicionar_nova_entrada():
    df = carregar_banco()

    st.markdown("### 📝 Adicionar nova demanda e resposta")

    sei = st.text_input("Nº do processo SEI")
    tipo = st.selectbox("Tipo do documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
    numero_doc = st.text_input("Nº do documento")
    autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
    texto_recebido = st.text_area("Texto do documento recebido (demanda ou perguntas)")
    resposta = st.text_area("Texto da resposta institucional enviada")

    if st.button("💾 Salvar registro"):
        if sei and tipo and numero_doc and autoria and texto_recebido and resposta:
            nova_linha = {
                "Nº do processo SEI": sei,
                "Tipo do documento": tipo,
                "Nº do documento": numero_doc,
                "Autoria": autoria,
                "Texto do documento recebido": texto_recebido,
                "Texto da resposta institucional enviada": resposta
            }
            df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
            salvar_banco(df)
            st.success("✅ Registro salvo com sucesso!")
        else:
            st.error("⚠️ Preencha todos os campos antes de salvar.")

def buscar_semelhantes():
    df = carregar_banco()
    if df.empty:
        st.warning("Nenhuma resposta cadastrada ainda.")
        return

    consulta = st.text_area("Digite o texto ou pergunta que deseja buscar:")

    if st.button("🔍 Buscar"):
        if not consulta.strip():
            st.error("Por favor, digite algo para buscar.")
            return

        embeddings_existentes = model.encode(df["Texto do documento recebido"].tolist(), convert_to_tensor=True)
        embedding_consulta = model.encode(consulta, convert_to_tensor=True)
        similaridades = util.cos_sim(embedding_consulta, embeddings_existentes)[0]

        top_indices = np.argsort(similaridades)[-5:][::-1]
        st.write("### Resultados mais semelhantes:")

        for i in top_indices:
            st.markdown(f"""
            **Nº do processo SEI:** {df.iloc[i]['Nº do processo SEI']}  
            **Tipo:** {df.iloc[i]['Tipo do documento']}  
            **Nº do documento:** {df.iloc[i]['Nº do documento']}  
            **Autoria:** {df.iloc[i]['Autoria']}  
            **Similaridade:** {similaridades[i]:.2f}  
            **Texto recebido:** {df.iloc[i]['Texto do documento recebido']}  
            **Resposta enviada:** {df.iloc[i]['Texto da resposta institucional enviada']}
            """)
            st.markdown("---")

def visualizar_e_editar():
    df = carregar_banco()
    if df.empty:
        st.warning("Nenhum registro encontrado ainda.")
        return

    termo_busca = st.text_input("🔍 Buscar por número, texto ou autoria:")
    if termo_busca:
        termo = termo_busca.lower()
        df_filtrado = df[df.apply(lambda row: termo in str(row).lower(), axis=1)]
    else:
        df_filtrado = df

    # ajustar índice para começar em 1
    df_filtrado = df_filtrado.reset_index(drop=True)
    df_filtrado.index += 1

    st.write(f"**Total de registros:** {len(df_filtrado)}")
    st.dataframe(df_filtrado)

    if st.button("🔄 Atualizar lista"):
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### ✏️ Editar registro existente")

    # montar lista de escolha (mostra Nº SEI — Nº documento)
    escolhas = df["Nº do processo SEI"].astype(str) + " — " + df["Nº do documento"].astype(str)
    escolha = st.selectbox("Selecione o registro para editar:", [""] + escolhas.tolist())

    if escolha:
        idx = escolhas[escolhas == escolha].index[0]
        registro = df.loc[idx]

        n_processo = st.text_input("Nº do processo SEI", registro["Nº do processo SEI"])
        tipo_doc = st.text_input("Tipo do documento", registro["Tipo do documento"])
        n_documento = st.text_input("Nº do documento", registro["Nº do documento"])
        autoria = st.text_input("Autoria", registro["Autoria"])
        texto_recebido = st.text_area("Texto do documento recebido", registro["Texto do documento recebido"])
        resposta_enviada = st.text_area("Texto da resposta institucional enviada", registro["Texto da resposta institucional enviada"])

        if st.button("💾 Atualizar registro"):
            df.loc[idx] = [n_processo, tipo_doc, n_documento, autoria, texto_recebido, resposta_enviada]
            salvar_banco(df)
            st.success("✅ Registro atualizado com sucesso!")
            st.experimental_rerun()

# ---------------- Login fixo simples ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("### 🔐 Acesso restrito à equipe DPL/ICMBio")
    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")
    if st.button("Entrar"):
        if usuario == "DPL" and senha == "ICMBio2025!":
            st.session_state.logged_in = True
            st.success("Login realizado com sucesso! ✅")
            st.experimental_rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")
else:
    menu = st.sidebar.radio("Menu", [
        "Adicionar nova demanda e resposta",
        "Buscar demandas semelhantes",
        "Visualizar demandas e respostas registradas",
        "Sair"
    ])

    if menu == "Adicionar nova demanda e resposta":
        adicionar_nova_entrada()
    elif menu == "Buscar demandas semelhantes":
        buscar_semelhantes()
    elif menu == "Visualizar demandas e respostas registradas":
        visualizar_e_editar()
    elif menu == "Sair":
        st.session_state.logged_in = False
        st.success("Logout realizado com sucesso.")
        st.experimental_rerun()

