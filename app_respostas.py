import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ----------------------------------------------------------
# CONFIGURAÇÕES VISUAIS DO APP
# ----------------------------------------------------------
st.set_page_config(
    page_title="Banco de Respostas da DPL",
    page_icon="🌿",
    layout="wide"
)

# CSS personalizado (cores institucionais e cabeçalho bonito)
st.markdown("""
    <style>
    body {
        background-color: #F9F9F6;
        color: #333333;
    }
    .main {
        background-color: #F9F9F6;
    }
    .stApp {
        background-color: #F9F9F6;
    }
    header[data-testid="stHeader"] {
        background-color: #1B5E20;
    }
    [data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }
    h1, h2, h3, h4 {
        color: #1B5E20;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# CABEÇALHO COM LOGO
# ----------------------------------------------------------
st.image("https://www.gov.br/icmbio/pt-br/assuntos/biodiversidade/unidade-de-conservacao/unidades-de-biomas/marinho/lista-de-ucs/parna-marinho-dos-abrolhos/fomulario-denuncia/icmbio-logo-1.png/@@images/93d85e33-e72b-423a-bc35-5d1b1f09b402.png", width=180)
st.title("Banco de Respostas da DPL")
st.caption("🌿 Harmonizando manifestações institucionais com inovação e gestão do conhecimento")

DATA_FILE = "banco_respostas.csv"

# ----------------------------------------------------------
# CARREGAMENTO DO MODELO SEMÂNTICO
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

# ----------------------------------------------------------
# FUNÇÃO PARA CARREGAR OU CRIAR BANCO
# ----------------------------------------------------------
def carregar_banco():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            "Nº do processo SEI",
            "Tipo do documento",
            "Nº do documento",
            "Autoria",
            "Texto do documento recebido",
            "Texto da resposta institucional enviada"
        ])

# ----------------------------------------------------------
# LOGIN FIXO
# ----------------------------------------------------------
if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    st.markdown("### 🔐 Acesso restrito à equipe DPL/ICMBio")

    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")

    if st.button("Entrar"):
        if usuario == "DPL" and senha == "ICMBio2025!":
            st.session_state.logado = True
            st.success("Arrasou! Login realizado com sucesso! ✅")
            st.rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")

else:
    # ===== MENU LATERAL =====
menu = st.sidebar.radio("Menu", ["Adicionar nova demanda e resposta",
                                 "Buscar demandas semelhantes",
                                 "Visualizar demandas e respostas registradas",
                                 "Sair"])

# ===== OPÇÃO 1: Adicionar nova demanda e resposta =====
if menu == "Adicionar nova demanda e resposta":
    st.subheader("Adicionar nova demanda e resposta")
    adicionar_nova_entrada()

# ===== OPÇÃO 2: Buscar demandas semelhantes =====
elif menu == "Buscar demandas semelhantes":
    st.subheader("Buscar demandas semelhantes")
    buscar_semelhantes()

# ===== NOVA OPÇÃO 3: Visualizar demandas =====
elif menu == "Visualizar demandas e respostas registradas":
    st.subheader("📋 Demandas e respostas registradas")

    # Verifica se existe o arquivo CSV
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)

        # Campo de busca
        termo_busca = st.text_input("🔍 Buscar por número, tipo ou autoria:")

        if termo_busca:
            termo = termo_busca.lower()
            df_filtrado = df[df.apply(lambda row: termo in str(row).lower(), axis=1)]
        else:
            df_filtrado = df

        # Mostra o total de registros
        st.write(f"**Total de registros:** {len(df_filtrado)}")

        # Mostra a tabela
        st.dataframe(df_filtrado)

        # Botão para atualizar
        if st.button("🔄 Atualizar lista"):
            st.rerun()
    else:
        st.warning("Nenhum registro encontrado ainda.")

# ===== OPÇÃO 4: Sair =====
elif menu == "Sair":
    st.session_state["logged_in"] = False
    st.success("Logout realizado com sucesso.")
    st.rerun()






