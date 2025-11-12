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
st.caption("🌿 Harmonizando manifestações institucionais
