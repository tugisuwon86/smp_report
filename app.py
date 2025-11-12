# app.py
import streamlit as st
import json
import pandas as pd
from utils import extract_text_from_pdf, build_prompt, query_openai, read_file_to_base64, call_gemini_api, convert_markdown_to_dataframe
from io import StringIO 

st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")
option = st.selectbox(
    "Select File Types",
    ("PDF", "Images"),
)

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
    
if option == 'Images':
    uploaded_file = st.file_uploader(
        "Upload your Image File (JPG or PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
        if st.button("Extract Data with Gemini", type="primary"):
            with st.spinner("Analyzing image and structuring data..."):
                try:
                    # Read the file data
                    base64_image, mime_type = read_file_to_base64(uploaded_file)
                    
                    # Call the Gemini API
                    extracted_data = call_gemini_api(base64_image, mime_type)
                    df_extracted = convert_markdown_to_dataframe(extracted_data)
                    
                    st.success("Extraction Complete!")
                    st.subheader("Extracted Structured Data (Markdown)")
                    
                    # Display the raw Markdown response
                    st.code(extracted_data, language="markdown")
                    
                    # Display the rendered Markdown table for a cleaner view
                    st.subheader("Rendered Table Preview")
                    st.markdown(extracted_data)
    
                    # Check if the DataFrame conversion was successful and it's a valid table
                    if not df_extracted.empty and 'Result' not in df_extracted.columns:
                         st.dataframe(df_extracted)
                         
                         # Optional: Add download button for CSV
                         csv = df_extracted.to_csv(index=False).encode('utf-8')
                         st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name='extracted_data.csv',
                            mime='text/csv',
                            key='download_csv',
                            type='secondary'
                         )

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
elif option == 'PDF':
    uploaded_files = st.file_uploader("Upload one or more PDF purchase orders", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Extracting data from {uploaded_file.name}..."):
                text = extract_text_from_pdf(uploaded_file)
                st.write(text)
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
