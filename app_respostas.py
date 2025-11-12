import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ----------------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# ----------------------------------------------------------
st.set_page_config(page_title="Banco de Respostas DPL - ICMBio", layout="wide")

DATA_FILE = "banco_respostas.csv"

# Carregar modelo semântico
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

# ----------------------------------------------------------
# FUNÇÃO PARA CARREGAR OU CRIAR BANCO DE DADOS
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
    st.title("🔐 Banco de Respostas DPL - ICMBio")

    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")

    if st.button("Entrar"):
        if usuario == "DPL" and senha == "ICMBio2025!":
            st.session_state.logado = True
            st.success("Login realizado com sucesso! ✅")
            st.experimental_rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")

else:
    # ------------------------------------------------------
    # MENU LATERAL
    # ------------------------------------------------------
    menu = st.sidebar.radio(
        "Menu principal",
        ["📥 Adicionar demanda/resposta", "🔍 Buscar semelhantes", "🚪 Sair"]
    )

    df = carregar_banco()

    # ------------------------------------------------------
    # OPÇÃO 1: ADICIONAR NOVA DEMANDA E RESPOSTA
    # ------------------------------------------------------
    if menu == "📥 Adicionar demanda/resposta":
        st.header("📥 Adicionar nova demanda e resposta")

        with st.form("add_form"):
            sei = st.text_input("Nº do processo SEI")
            tipo = st.selectbox("Tipo do documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
            numero_doc = st.text_input("Nº do documento")
            autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
            texto_demanda = st.text_area("Texto do documento recebido (demanda ou perguntas)")
            texto_resposta = st.text_area("Texto da resposta institucional enviada")
            submitted = st.form_submit_button("Salvar no banco")

            if submitted:
                nova_linha = pd.DataFrame([{
                    "Nº do processo SEI": sei,
                    "Tipo do documento": tipo,
                    "Nº do documento": numero_doc,
                    "Autoria": autoria,
                    "Texto do documento recebido": texto_demanda,
                    "Texto da resposta institucional enviada": texto_resposta
                }])
                df = pd.concat([df, nova_linha], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.success("✅ Demanda e resposta salvas com sucesso!")

    # ------------------------------------------------------
    # OPÇÃO 2: BUSCAR DEMANDAS SEMELHANTES
    # ------------------------------------------------------
    elif menu == "🔍 Buscar semelhantes":
        st.header("🔍 Buscar demandas semelhantes")

        consulta = st.text_area("Digite o texto ou pergunta que deseja verificar:")

        if st.button("Buscar no banco"):
            if len(df) == 0:
                st.warning("O banco de dados ainda está vazio.")
            else:
                consulta_emb = model.encode(consulta, convert_to_tensor=True)
                textos = df["Texto do documento recebido"].tolist()
                embeddings = model.encode(textos, convert_to_tensor=True)
                similaridades = util.pytorch_cos_sim(consulta_emb, embeddings)[0].cpu().numpy()

                top_k = np.argsort(similaridades)[::-1][:5]

                st.write("### Resultados mais semelhantes:")
                for i in top_k:
                    st.markdown(f"""
                    **Similaridade:** {similaridades[i]*100:.2f}%  
                    **Nº SEI:** {df.iloc[i]['Nº do processo SEI']}  
                    **Tipo:** {df.iloc[i]['Tipo do documento']}  
                    **Nº do documento:** {df.iloc[i]['Nº do documento']}  
                    **Autoria:** {df.iloc[i]['Autoria']}  
                    **Texto recebido:** {df.iloc[i]['Texto do documento recebido']}  
                    **Resposta institucional:** {df.iloc[i]['Texto da resposta institucional enviada']}  
                    ---
                    """)

    # ------------------------------------------------------
    # OPÇÃO 3: SAIR
    # ------------------------------------------------------
    elif menu == "🚪 Sair":
        st.session_state.logado = False
        st.success("Sessão encerrada com sucesso.")
        st.experimental_rerun()
