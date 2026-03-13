# app.py
import streamlit as st
import json
import pandas as pd
from utils import extract_text_from_pdf, build_prompt, query_openai, read_file_to_base64, call_gemini_api, convert_markdown_to_dataframe
from io import StringIO 
import base64

st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")
option = st.selectbox(
    "Select File Types",
    ("PDF", "Images"),
)

def download_button(data, filename, label):
    if isinstance(data, str):
        data = data.encode()

    b64 = base64.b64encode(data).decode()

    href = (
        '<a download="' + filename + '" '
        'href="data:application/octet-stream;base64,' + b64 + '" '
        'style="display:inline-block;padding:0.5em 1em;color:white;'
        'background-color:#FF4B4B;text-decoration:none;border-radius:0.5em;font-weight:600;">'
        + label +
        '</a>'
    )
    st.markdown(href, unsafe_allow_html=True)

def parser(content):

    # content = json.loads(content)
    
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
                    df = convert_markdown_to_dataframe(extracted_data)
                    
                    st.success("Extraction Complete!")
                    st.subheader("Extracted Structured Data (Markdown)")
                    
                    # Display the raw Markdown response
                    st.code(extracted_data, language="markdown")
                    
                    # Display the rendered Markdown table for a cleaner view
                    st.subheader("Rendered Table Preview")
                    st.markdown(extracted_data)
    
                    # Check if the DataFrame conversion was successful and it's a valid table
                    if not df.empty and 'Result' not in df.columns:
                         st.dataframe(df)
                         
                         # Optional: Add download button for CSV
                         csv = df.to_csv(index=False).encode('utf-8')
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
                prompt = build_prompt(text)
                response = query_openai(prompt)
                # st.write(type(response))
                # st.write(response)
                # try:
                vendor, ship_to, df = parser(response)
                st.subheader(f"📦 {uploaded_file.name}")
                st.write('Vendor Information')
                st.write(vendor)

                st.write('Shipping Information')
                st.write(ship_to)

                if ship_to["company"] != "SMP Corporation":
                    vendor_ = ship_to["company"]
                else:
                    vendor_ = vendor["company"]

                st.write('Item Information')
                st.dataframe(df)
                # except json.JSONDecodeError:
                #     st.error("Could not parse structured JSON. Here’s the raw response:")
                #     st.text(response)

# IIF generation (only if we have matched rows)

if not df.empty:
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    # Prepare CSV once
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = df.to_csv(index=False)

    with col1:
        download_button(
            df.to_csv(index=False),
            "output.csv",
            "Download CSV"
        )     
        
    from pages._utils import generate_purchase_order_iif, generate_sales_order_iif, load_qb_lists_from_iif, validate_items_against_qb

    @st.cache_data
    def load_qb_items():
        items, vendors, customers = load_qb_lists_from_iif("pages/smp.IIF")
        return items, vendors, customers

    qb_items, qb_vendors, qb_customers = load_qb_items()

    missing_items = validate_items_against_qb(matched_rows, qb_items)

    if missing_items:
        st.warning(
            f"⚠️ Warning: {len(missing_items)} item(s) not found in QuickBooks:\n"
            + "\n".join(missing_items[:5])
            + (f"\n... and {len(missing_items)-5} more" if len(missing_items) > 5 else "")
        )
    else:
        st.success("✅ All items validated against QuickBooks.")

    vendor_map = {
        "Geoshield": "Geoshield",
        "Hitek": "Hitek",
        "UVIRON": "UVIRON",
        "SMP": "SMP"
    }

    vendor_name = vendor_map.get(vendor_, vendor_)

    # Generate files once and store
    if "po_iif" not in st.session_state:
        st.session_state.po_iif = generate_purchase_order_iif(
            matched_rows,
            qb_items,
            vendor_name=vendor_name,
            container=False
        )

    if "so_iif" not in st.session_state:
        st.session_state.so_iif = generate_sales_order_iif(
            matched_rows,
            qb_items,
            customer_name="Default Customer",
            container=False
        )

    with col2:
        download_button(
            st.session_state.po_iif,
            f"purchase_order_{option_company}.iif",
            "Download PO (.iif)"
        )
    
    with col3:
        download_button(
            st.session_state.so_iif,
            f"sales_order_{option_company}.iif",
            "Download SO (.iif)"
        )
