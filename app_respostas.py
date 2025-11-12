# Formulário para adicionar nova resposta
st.subheader("Adicionar nova demanda e resposta")

with st.form("nova_resposta"):
    processo = st.text_input("Nº do processo SEI")
    tipo_doc = st.selectbox("Tipo do documento", ["Ofício", "Requerimento de Informação", "Indicação", "Outro"])
    numero_doc = st.text_input("Nº do documento")
    autoria = st.text_input("Autoria (ex: Dep. Federal João Silva - PT/SP)")
    texto_pergunta = st.text_area("Texto do documento recebido (demanda ou perguntas)")
    texto_resposta = st.text_area("Texto da resposta institucional enviada")
    submitted = st.form_submit_button("Salvar")

if submitted:
    nova_linha = {
        "Nº do processo SEI": processo,
        "Tipo do documento": tipo_doc,
        "Nº do documento": numero_doc,
        "Autoria": autoria,
        "Texto do documento recebido": texto_pergunta,
        "Texto da resposta institucional enviada": texto_resposta
    }
    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    st.success("✅ Demanda e resposta adicionadas com sucesso!")
