# app.py
import streamlit as st
import json
import pandas as pd
from utils import extract_text_from_pdf, build_prompt, query_openai

st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")

uploaded_files = st.file_uploader("Upload one or more PDF purchase orders", type=["pdf"], accept_multiple_files=True)

def parser(content):

    content = json.loads(content)
    
    vendor = {}
    ## customer level - bill/vendor
    bill_to_customer = content['bill_to_customer']
    vendor['company'] = bill_to_customer['company_name']
    vendor['address'] = bill_to_customer['address']
    vendor['email'] = bill_to_customer['email']
    vendor['phone'] = bill_to_customer['phone']

    ## shipping
    ship_to = {}
    ship_to_customer = content['ship_to_customer']
    ship_to['company'] = ship_to_customer['company_name']
    ship_to['address'] = ship_to_customer['address']
    ship_to['email'] = ship_to_customer['email']
    ship_to['phone'] = ship_to_customer['phone']

    df = pd.DataFrame(content['items'])
    return vendor, ship_to, df



if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Extracting data from {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            prompt = build_prompt(text)
            response = query_openai(prompt)
            # st.write(type(response))
            # st.write(response)
            try:
                vendor, ship_to, df = parser(response.content)
                st.subheader(f"📦 {uploaded_file.name}")
                st.write('Vendor Information')
                st.write(vendor)

                st.write('Shipping Information')
                st.write(ship_to)

                st.write('Item Information')
                st.dataframe(df)
            except json.JSONDecodeError:
                st.error("Could not parse structured JSON. Here’s the raw response:")
                st.text(response)
