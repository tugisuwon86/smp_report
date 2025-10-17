# app.py
import streamlit as st
import json
import pandas as pd
from utils import extract_text_from_pdf, query_mistral, build_prompt, query_openai

st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")

uploaded_files = st.file_uploader("Upload one or more PDF purchase orders", type=["pdf"], accept_multiple_files=True)

def parser(response):
    content = response['content']

    vendor = {}
    ## customer level - bill/vendor
    vendor['company'] = content['bill_to_customer']['company_name']
    vendor['address'] = content['bill_to_customer']['address']
    vendor['email'] = content['bill_to_customer']['email']
    vendor['phone'] = content['bill_to_customer']['phone']

    ## shipping
    ship_to = {}
    ship_to['company'] = content['bill_to_customer']['company_name']
    ship_to['address'] = content['bill_to_customer']['address']
    ship_to['email'] = content['bill_to_customer']['email']
    ship_to['phone'] = content['bill_to_customer']['phone']

    df = pd.DataFrame(content['items'])
    return vendor, ship_to, df



if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Extracting data from {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            prompt = build_prompt(text)
            response = query_openai(prompt)
            st.write(response.keys())
            st.write(response)
            # try:
            #     vendor, ship_to, df = parser(response)
            #     st.subheader(f"📦 {uploaded_file.name}")
            #     st.write('Vendor Information')
            #     st.write(vendor)

            #     st.write('Shipping Information')
            #     st.write(ship_to)

            #     st.write('Item Information')
            #     st.dataframe(df)
            # except json.JSONDecodeError:
            #     st.error("Could not parse structured JSON. Here’s the raw response:")
            #     st.text(response)
