# app_respostas.py  (versão final com Google Sheets backend estável)
import streamlit as st
import pandas as pd
import os
import json
import gspread
from gspread_dataframe import set_with_dataframe
from sentence_transformers import SentenceTransformer, util
import numpy as np

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

# -----------------------------------------------------------------
# 🔥 CONFIGURAÇÃO E AUTENTICAÇÃO
# -----------------------------------------------------------------

secrets = st.secrets.to_dict()

# Verifica as chaves essenciais
if "gcp_service_account" not in secrets or "sheet_url" not in secrets:
    st.error("❌ Erro: As chaves essenciais (gcp_service_account ou sheet_url) não estão definidas no Streamlit Secrets.")
    st.stop()

# Pega credenciais de login e Sheets
service_account_info = secrets["gcp_service_account"]
sheet_url = secrets["sheet_url"]

# Pega credenciais de login
LOGIN_USER = st.secrets.get("LOGIN_USER", "DPL_DEFAULT")
LOGIN_PASS = st.secrets.get("LOGIN_PASS", "DEFAULT_PASS")


# -----------------------------------------------------------------
# Autenticar gspread
# -----------------------------------------------------------------
try:
    # 🌟 CORREÇÃO CRÍTICA: Substitui as quebras de linha literais pelo caractere real \n.
    # Isso resolve o erro "Read 105 bytes instead of expected 29" ao ler a chave privada.
    service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")
    
    gc = gspread.service_account_from_dict(service_account_info)
    SHEET = gc.open_by_url(sheet_url)
    ws = SHEET.sheet1
except Exception as e:
    st.error(f"❌ Erro ao conectar ao Google Sheets. Verifique a URL e as permissões de compartilhamento. Detalhes: {e}")
    st.stop()

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

@st.cache_data(ttl=600) # Cache de 10 minutos para o banco
def carregar_banco():
    try:
        records = ws.get_all_records() 
        if not records:
            return pd.DataFrame(columns=COLS)
        
        df = pd.DataFrame.from_records(records, dtype=str)
        
        for c in COLS:
            if c not in df.columns:
                df[c] = ""
        
        df = df[COLS]
        # Adiciona um ID de linha para facilitar a busca interna
        df['ID_Linha'] = df.index
        return df
    except Exception as e:
        st.error(f"Erro ao carregar planilha: {e}.")
        return pd.DataFrame(columns=COLS)

def salvar_banco(df):
    try:
        # Remove a coluna ID_Linha antes de salvar de volta no Sheets, se existir
        df_to_save = df.drop(columns=['ID_Linha'], errors='ignore')
        
        ws.clear()
        set_with_dataframe(ws, df_to_save, include_index=False, include_column_header=True)
        # Limpa o cache após salvar para forçar uma nova leitura na próxima vez
        carregar_banco.clear() 
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Erro ao salvar na planilha: {e}")

# ---------------- Funções do app ----------------
def adicionar_nova_entrada():
    st.markdown("### 📝 Adicionar nova demanda e resposta")

    with st.form("form_nova_entrada"):
        sei = st.text_input("Nº do processo SEI")
        tipo = st.selectbox("Tipo do documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
        numero_doc = st.text_input("Nº do documento")
        autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
        texto_recebido = st.text_area("Texto do documento recebido (demanda ou perguntas)")
        resposta = st.text_area("Texto da resposta institucional enviada")

        salvar_button = st.form_submit_button("💾 Salvar registro")
    
    if salvar_button:
        if sei and tipo and numero_doc and autoria and texto_recebido and resposta:
            df = carregar_banco()
            nova_linha = {
                "Nº do processo SEI": sei,
                "Tipo do documento": tipo,
                "Nº do documento": numero_doc,
                "Autoria": autoria,
                "Texto do documento recebido": texto_recebido,
                "Texto da resposta institucional enviada": resposta
            }
            # Concatena a nova linha
            df_sem_id = df.drop(columns=['ID_Linha'], errors='ignore')
            df = pd.concat([df_sem_id, pd.DataFrame([nova_linha])], ignore_index=True)
            salvar_banco(df)
            st.success("✅ Registro salvo com sucesso! Atualizando lista...")
        else:
            st.error("⚠️ Preencha todos os campos antes de salvar.")

def buscar_semelhantes():
    df = carregar_banco()
    if df.empty or len(df) == 0:
        st.warning("Nenhuma resposta cadastrada ainda. Por favor, adicione registros primeiro.")
        return

    st.markdown("### 🔍 Buscar demandas semelhantes (IA Semântica)")
    consulta = st.text_area("Digite o texto ou pergunta que deseja buscar:")

    if st.button("🔍 Buscar"):
        if not consulta.strip():
            st.error("Por favor, digite algo para buscar.")
            return
        
        df_para_analise = df[df["Texto do documento recebido"].astype(str).str.strip() != ""]
        if df_para_analise.empty:
            st.warning("Nenhum registro com 'Texto do documento recebido' válido para análise.")
            return

        with st.spinner("Analisando similaridades..."):
            embeddings_existentes = model.encode(df_para_analise["Texto do documento recebido"].tolist(), convert_to_tensor=True)
            embedding_consulta = model.encode(consulta, convert_to_tensor=True)
            similaridades = util.cos_sim(embedding_consulta, embeddings_existentes)[0]

        top_indices = np.argsort(similaridades)[-5:][::-1]
        st.write("### Resultados mais semelhantes:")

        for i in top_indices:
            registro = df_para_analise.iloc[i]
            st.markdown(f"""
            <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            **Similaridade:** **{similaridades[i]:.2f}**
            **Nº do processo SEI:** {registro['Nº do processo SEI']} | **Tipo:** {registro['Tipo do documento']} | **Nº do documento:** {registro['Nº do documento']}  
            **Autoria:** {registro['Autoria']}  
            **Demanda Recebida:** *{registro['Texto do documento recebido']}* **Resposta Institucional Enviada:** {registro['Texto da resposta institucional enviada']}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

def visualizar_e_editar():
    df = carregar_banco()
    if df.empty or len(df) == 0:
        st.warning("Nenhum registro encontrado ainda. Por favor, adicione registros primeiro.")
        return

    st.markdown("### 🔍 Visualizar e Editar registros")
    
    # ------------------ Busca e Visualização ------------------
    termo_busca = st.text_input("🔍 Buscar por número, texto ou autoria:", key="busca_view_edit")
    if termo_busca:
        termo = termo_busca.lower()
        df_filtrado = df[df.apply(lambda row: termo in str(row).lower(), axis=1)]
    else:
        df_filtrado = df

    df_para_exibir = df_filtrado.drop(columns=['ID_Linha'], errors='ignore').copy()
    df_para_exibir.index = np.arange(1, len(df_para_exibir) + 1)
    st.write(f"**Total de registros filtrados:** {len(df_para_exibir)}")
    st.dataframe(df_para_exibir)
    
    st.markdown("---")
    
    # ------------------ Edição de Registro ------------------
    st.markdown("### ✏️ Editar registro existente")

    escolhas = df["Nº do processo SEI"].astype(str) + " — " + df["Nº do documento"].astype(str)
    
    df['Escolha'] = escolhas # Adiciona a coluna temporária para facilitar a busca
    
    escolha_selecionada = st.selectbox("Selecione o registro para editar:", [""] + escolhas.tolist(), key="select_edit")

    if escolha_selecionada:
        registro_a_editar = df[df['Escolha'] == escolha_selecionada].iloc[0]
        idx_real = registro_a_editar.name 
        
        st.markdown(f"**Registro selecionado:** `{escolha_selecionada}`")

        with st.form("form_edicao"):
            n_processo = st.text_input("Nº do processo SEI", registro_a_editar["Nº do processo SEI"])
            tipo_doc = st.text_input("Tipo do documento", registro_a_editar["Tipo do documento"])
            n_documento = st.text_input("Nº do documento", registro_a_editar["Nº do documento"])
            autoria = st.text_input("Autoria", registro_a_editar["Autoria"])
            texto_recebido = st.text_area("Texto do documento recebido", registro_a_editar["Texto do documento recebido"])
            resposta_enviada = st.text_area("Texto da resposta institucional enviada", registro_a_editar["Texto da resposta institucional enviada"])

            if st.form_submit_button("💾 Atualizar registro"):
                df.loc[idx_real, COLS] = [n_processo, tipo_doc, n_documento, autoria, texto_recebido, resposta_enviada]
                
                # A função salvar_banco cuida do drop da coluna 'ID_Linha' e 'Escolha' (implicitamente)
                salvar_banco(df)
                st.success("✅ Registro atualizado com sucesso! Atualizando lista...")
                
    if 'Escolha' in df.columns:
        df.drop(columns=['Escolha'], inplace=True) 

    if st.button("🔄 Recarregar dados da planilha"):
        carregar_banco.clear()
        st.experimental_rerun()


# ---------------- Login ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("### 🔐 Acesso restrito à equipe DPL/ICMBio")
    usuario = st.text_input("Usuário:")
    senha = st.text_input("Senha:", type="password")
    
    if st.button("Entrar"):
        if usuario == LOGIN_USER and senha == LOGIN_PASS:
            st.session_state.logged_in = True
            st.success("Login realizado com sucesso! ✅")
            st.experimental_rerun()
        else:
            st.error("❌ Usuário ou senha incorretos.")
else:
    # ---------------- Menu Principal ----------------
    st.sidebar.markdown(f"**Usuário:** `{LOGIN_USER}`")
    menu = st.sidebar.radio("Menu", [
        "Buscar demandas semelhantes", 
        "Adicionar nova demanda e resposta",
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
        st.sidebar.success("Logout realizado com sucesso.")
        st.experimental_rerun()
