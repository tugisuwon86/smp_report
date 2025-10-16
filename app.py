# app.py
import streamlit as st
import json
from utils import extract_text_from_pdf, query_mistral, build_prompt

st.write(st.secrets['hugging-face'])
HF_TOKEN = st.secrets['hugging-face']['api-token']
st.write(HF_TOKEN)
st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")

uploaded_files = st.file_uploader("Upload one or more PDF purchase orders", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Extracting data from {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            prompt = build_prompt(text)
            response = query_mistral(prompt)

            try:
                parsed = json.loads(response)
                st.subheader(f"📦 {uploaded_file.name}")
                st.json(parsed)
            except json.JSONDecodeError:
                st.error("Could not parse structured JSON. Here’s the raw response:")
                st.text(response)
