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
st.image(
    "https://www.gov.br/icmbio/pt-br/assuntos/biodiversidade/unidade-de-conservacao/unidades-de-biomas/marinho/lista-de-ucs/parna-marinho-dos-abrolhos/fomulario-denuncia/icmbio-logo-1.png/@@images/93d85e33-e72b-423a-bc35-5d1b1f09b402.png",
    width=180
)
st.title("Banco de Respostas da DPL - ICMBio")
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
# FUNÇÃO: ADICIONAR NOVA ENTRADA
# ----------------------------------------------------------
def adicionar_nova_entrada():
    st.markdown("### 📝 Adicionar nova demanda/resposta")
    df = carregar_banco()

    n_processo = st.text_input("Nº do processo SEI")
    tipo_doc = st.selectbox("Tipo do documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
    n_documento = st.text_input("Nº do documento")
    autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
    texto_recebido = st.text_area("Texto do documento recebido (demanda ou perguntas)")
    resposta_enviada = st.text_area("Texto da resposta institucional enviada")

    if st.button("💾 Salvar registro"):
        if n_processo and tipo_doc and autoria and texto_recebido and resposta_enviada:
            novo_registro = pd.DataFrame([{
                "Nº do processo SEI": n_processo,
                "Tipo do documento": tipo_doc,
                "Nº do documento": n_documento,
                "Autoria": autoria,
                "Texto do documento recebido": texto_recebido,
                "Texto da resposta institucional enviada": resposta_enviada
            }])
            df = pd.concat([df, novo_registro], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("✅ Registro salvo com sucesso!")
        else:
            st.warning("⚠️ Por favor, preencha todos os campos obrigatórios.")

# ----------------------------------------------------------
# FUNÇÃO: BUSCAR SEMELHANTES
# ----------------------------------------------------------
def buscar_semelhantes():
    st.markdown("### 🔍 Buscar demandas semelhantes")
    df = carregar_banco()

    if df.empty:
        st.info("Ainda não há registros para comparar.")
        return

    consulta = st.text_area("Cole aqui o texto da nova demanda para buscar semelhanças:")
    if st.button("Buscar"):
        if consulta.strip():
            embeddings = model.encode(df["Texto do documento recebido"].tolist(), convert_to_tensor=True)
            consulta_emb = model.encode(consulta, convert_to_tensor=True)
            similaridades = util.pytorch_cos_sim(consulta_emb, embeddings)[0]
            df["Similaridade"] = similaridades.cpu().numpy()
            resultados = df.sort_values(by="Similaridade", ascending=False).head(5)
            st.write("### Resultados mais semelhantes:")
            st.dataframe(resultados[[
                "Nº do processo SEI",
                "Tipo do documento",
                "Nº do documento",
                "Autoria",
                "Texto do documento recebido",
                "Texto da resposta institucional enviada",
                "Similaridade"
            ]])
        else:
            st.warning("Digite um texto para buscar semelhanças.")

# ----------------------------------------------------------
# FUNÇÃO: VISUALIZAR E EDITAR REGISTROS
# ----------------------------------------------------------
def visualizar_e_editar():
    st.subheader("📋 Demandas e respostas registradas")
    if not os.path.exists(DATA_FILE):
        st.warning("Nenhum registro encontrado ainda.")
        return

    df = pd.read_csv(DATA_FILE)

    termo_busca = st.text_input("🔍 Buscar por número, texto ou autoria:")
    if termo_busca:
        termo = termo_busca.lower()
        df_filtrado = df[df.apply(lambda row: termo in str(row).lower(), axis=1)]
    else:
        df_filtrado = df

    st.write(f"**Total de registros:** {len(df_filtrado)}")
    st.dataframe(df_filtrado)

    # Botão para atualizar
    if st.button("🔄 Atualizar lista"):
        st.rerun()

    # Se quiser editar um registro
    st.markdown("---")
    st.markdown("### ✏️ Editar registro existente")

    if len(df) > 0:
        opcoes = df["Nº do processo SEI"].astype(str) + " — " + df["Nº do documento"].astype(str)
        escolha = st.selectbox("Selecione o registro para editar:", [""] + opcoes.tolist())

        if escolha:
            idx = opcoes[opcoes == escolha].index[0]
            registro = df.loc[idx]

            n_processo = st.text_input("Nº do processo SEI", registro["Nº do processo SEI"])
            tipo_doc = st.text_input("Tipo do documento", registro["Tipo do documento"])
            n_documento = st.text_input("Nº do documento", registro["Nº do documento"])
            autoria = st.text_input("Autoria", registro["Autoria"])
            texto_recebido = st.text_area("Texto do documento recebido", registro["Texto do documento recebido"])
            resposta_enviada = st.text_area("Texto da resposta institucional enviada", registro["Texto da resposta institucional enviada"])

            if st.button("💾 Atualizar registro"):
                df.loc[idx] = [n_processo, tipo_doc, n_documento, autoria, texto_recebido, resposta_enviada]
                df.to_csv(DATA_FILE, index=False)
                st.success("✅ Registro atualizado com sucesso!")
                st.rerun()

# ----------------------------------------------------------
# LOGIN FIXO
# ----------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("### 🔐 Acesso restrito à equipe DPL/ICMBio")

    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")

    if st.button("Entrar"):
        if usuario == "DPL" and senha == "ICMBio2025!":
            st.session_state.logged_in = True
            st.success("Arrasou! Login realizado com sucesso! ✅")
            st.rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")
else:
    # ===== MENU LATERAL =====
    menu = st.sidebar.radio("Menu", [
        "Adicionar nova demanda/resposta",
        "Buscar demandas semelhantes",
        "Visualizar demandas/respostas registradas",
        "Sair"
    ])

    if menu == "Adicionar nova demanda/resposta":
        adicionar_nova_entrada()

    elif menu == "Buscar demandas semelhantes":
        buscar_semelhantes()

    elif menu == "Visualizar demandas/respostas registradas":
        visualizar_e_editar()

    elif menu == "Sair":
        st.session_state.logged_in = False
        st.success("Logout realizado com sucesso.")
        st.rerun()

