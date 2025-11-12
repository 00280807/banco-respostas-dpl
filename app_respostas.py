# Banco de Respostas DPL - ICMBio (login único da equipe)
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

# ---------- CONFIGURAÇÕES INICIAIS ----------
st.set_page_config(page_title="Banco de Respostas DPL - ICMBio", layout="wide")

# ---------- LOGIN ÚNICO ----------
USUARIO_CORRETO = "DPL"
SENHA_CORRETA = "ICMBio2025!"

if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    st.title("🔒 Acesso Restrito - Banco de Respostas DPL - ICMBio")
    st.write("Insira o login da equipe para continuar:")

    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")

    if st.button("Entrar"):
        if usuario == USUARIO_CORRETO and senha == SENHA_CORRETA:
            st.session_state.logado = True
            st.success("✅ Acesso autorizado! Carregando o sistema...")
            st.rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")
    st.stop()

# ---------- SISTEMA PRINCIPAL ----------
st.title("🌿 Banco de Respostas DPL - ICMBio")
st.caption(f"Usuário logado: {USUARIO_CORRETO}")

# Carrega o modelo de linguagem (para comparar textos)
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo = carregar_modelo()

ARQUIVO_DADOS = "respostas.csv"

def carregar_dados():
    if os.path.exists(ARQUIVO_DADOS):
        return pd.read_csv(ARQUIVO_DADOS)
    else:
        return pd.DataFrame(columns=[
            "processo_sei", "tipo_documento", "autoria",
            "texto_pergunta", "texto_resposta", "embedding_pergunta"
        ])

def salvar_dados(df):
    df.to_csv(ARQUIVO_DADOS, index=False)

def gerar_embedding(texto):
    return modelo.encode(texto)

# Menu lateral
aba = st.sidebar.radio("Escolha uma opção:", ["Cadastrar novo registro", "Buscar demandas semelhantes", "Sair"])

df = carregar_dados()

# ---------- ABA 1: CADASTRAR NOVO REGISTRO ----------
if aba == "Cadastrar novo registro":
    st.subheader("📝 Cadastrar nova demanda e resposta")

    processo = st.text_input("Nº do processo SEI")
    tipo = st.selectbox("Tipo de documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
    autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
    texto_pergunta = st.text_area("📩 Texto do documento recebido (demanda ou pergunta)")
    texto_resposta = st.text_area("📤 Texto da resposta institucional enviada")

    if st.button("💾 Salvar no banco"):
        if not processo or not texto_pergunta or not texto_resposta:
            st.warning("Preencha todos os campos obrigatórios.")
        else:
            embedding = gerar_embedding(texto_pergunta)
            novo = pd.DataFrame({
                "processo_sei": [processo],
                "tipo_documento": [tipo],
                "autoria": [autoria],
                "texto_pergunta": [texto_pergunta],
                "texto_resposta": [texto_resposta],
                "embedding_pergunta": [embedding.tolist()]
            })
            df = pd.concat([df, novo], ignore_index=True)
            salvar_dados(df)
            st.success(f"Registro do processo {processo} salvo com sucesso! ✅")

# ---------- ABA 2: BUSCAR DEMANDAS SEMELHANTES ----------
if aba == "Buscar demandas semelhantes":
    st.subheader("🔍 Buscar demandas parecidas no banco")

    consulta = st.text_area("Cole aqui o texto do novo documento recebido (pergunta, ofício, etc.)")

    top_k = st.slider("Quantos resultados parecidos deseja ver?", 1, 10, 3)

    if st.button("Buscar"):
        if len(df) == 0:
            st.warning("O banco de respostas ainda está vazio.")
        elif not consulta.strip():
            st.warning("Cole um texto para buscar.")
        else:
            consulta_emb = gerar_embedding(consulta)
            embeddings_salvos = np.vstack(df["embedding_pergunta"].apply(eval).values)
            similaridades = util.cos_sim(consulta_emb, embeddings_salvos)[0]
            indices = np.argsort(similaridades)[::-1][:top_k]

            st.markdown("### Resultados mais semelhantes:")

            for i in indices:
                st.write(f"**Processo SEI:** {df.iloc[i]['processo_sei']}")
                st.write(f"**Tipo:** {df.iloc[i]['tipo_documento']}")
                st.write(f"**Autoria:** {df.iloc[i]['autoria']}")
                st.write(f"**Similaridade:** {similaridades[i]:.2f}")
                with st.expander("📩 Ver texto da pergunta recebida"):
                    st.write(df.iloc[i]['texto_pergunta'])
                with st.expander("📤 Ver resposta institucional dada"):
                    st.write(df.iloc[i]['texto_resposta'])
                st.markdown("---")

# ---------- ABA 3: SAIR ----------
if aba == "Sair":
    st.session_state.logado = False
    st.experimental_rerun()
