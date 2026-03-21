# app.py
import streamlit as st
import json
import pandas as pd
from utils import extract_text_from_pdf, build_prompt, query_openai, read_file_to_base64, call_gemini_api, convert_markdown_to_dataframe
from io import StringIO 
import base64
import re

st.set_page_config(page_title="Purchase Order Extractor", layout="wide")
st.title("📄 Purchase Order Extractor")
option = st.selectbox(
    "Select File Types",
    ("PDF", "Images", "Excel"),
)

def excel_to_text(df):
    """
    Convert dataframe to structured text for LLM
    """
    return df.to_csv(index=False)

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

from rapidfuzz import fuzz

def extract_width_from_meta(desc):
    # Example: 'Megamax 20% 40"X100' → 40
    m = re.search(r'(\d+)\s*["]?X', desc)
    if m:
        return int(m.group(1))
    return None
    
def best_meta_match(row, meta_df, option_company):
    debugging = False
    if debugging:
        st.write("Here processing: ", row)
    row["composition"] = "nan"
    item = str(row["product"] + ' ' + row["description"])
    if option_company == "Hitek":
        if 'ceramic ir' in item.lower():
            item += ' PREMIUM'
        elif 'plus' in item.lower() and 'ceramic' in item.lower():
            item += ' ALPHA'
        
    try:
        vlt = float(str(row["vlt"]).strip().replace('%', ''))
    except:
        if option_company == 'Geoshield':
            vlt = 'PPF'
            item += ' PPF'
        else:
            vlt = 0
    if option_company == 'Geoshield' and vlt == 75:
        vlt = 70 # just for geoshield
    width_final = int(row["width"])

    # if width = 12 
    factor = 1
    # if width_final == 12:
    #     row["composition"] = "5*12"
    #     width_final = 60
    #     factor = 5

    # 1️⃣ Filter by matching VLT
    candidates = meta_df[meta_df["VLT"] == vlt]
    if candidates.shape[0] == 0:
        candidates = meta_df.copy()

    candidates["compare"] = candidates[["QB Description", "Width"]].apply(lambda x: str(x[0]) + ' ' + str(x[1]), axis=1)
    candidates = candidates[candidates["compare"].str.contains(str(width_final))]
    # st.dataframe(candidates.head(2))
    if candidates.empty:
        return None

    best_score = -1
    best_row = None

    # force width to be 60 if composition exists!
    if str(row["composition"]) != 'nan' and ('/' in str(row["composition"]) or '*' in str(row["composition"])):
        row["width"] = 60
    if str(row["composition"]) != 'nan' and '/' not in str(row["composition"]) and '*' not in str(row["composition"]):
        row["composition"] = 'nan'
        
    for _, m in candidates.iterrows():
        # make sure slitting/composition is found under width slitting
        if str(row["composition"]) != 'nan' and str(row["composition"]) not in str(m["Width Slitting"]):
            # st.write('composition not found :', row["composition"], m["Width Slitting"])
            continue
        # make sure the width match!
        if str(row["width"])+"\"" not in m["QB Description"]:
            continue
        if debugging:
            st.write("description value: ", m["QB Description"], m["Description"], item, any([x.lower() in m["QB Description"].lower() for x in item.split()]))
        if any([x.lower() in m["QB Description"].lower() for x in item.split()]) or any([x.lower() in m["Description"].lower() for x in item.split()]):
            total_score = -1
            multiplier = sum([[0,1][x.lower() in m["QB Description"].lower() or x.lower() in m["Description"]] for x in item.split()])
            # if factor != 1:
            #     st.write('special case: ', m["Width"], row["composition"], ('/' in str(m["Width"]) or '*' in str(m["Width"])), str(row["composition"]) != 'nan', (str(row["composition"]) != 'nan' and ('/' in str(m["Width"]) or '*' in str(m["Width"]))))
            if (str(row["composition"]) == 'nan' and '/' not in str(m["Width"])) or (str(row["composition"]) != 'nan' and ('/' in str(m["Width Slitting"]) or '*' in str(m["Width Slitting"]))):
                meta_width = extract_width_from_meta(m["Description"])
                #st.write('meta_width: ', meta_width)
                # 2️⃣ Width match (only when width_final < 60)
                if width_final == 'PPF' and width_final in m["Description"]:
                    width_score = 100
                elif width_final < 60 and meta_width == width_final:
                    width_score = 100
                else:
                    width_score = 0

    
                # 3️⃣ Fuzzy match on item name
                # score1 = fuzz.token_set_ratio(item, m["description"])
                score1 = max(fuzz.token_set_ratio(str(row["composition"]), str(m["Width"])), fuzz.token_set_ratio(str(row["composition"]), str(m["QB Description"])))
                score2 = max(fuzz.token_set_ratio(item, m["QB Description"]), fuzz.token_set_ratio(item, m["Description"]))
                item_score = score1 + score2
                total_score = width_score + item_score * multiplier
                if debugging:
                    st.write(width_score, item_score, total_score)
            if total_score > best_score:
                # st.write(item, m["Width"], m["Description"], total_score)
                best_score = total_score
                best_row = m

    return best_row, factor

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

option_company = st.selectbox(
    "Company Name: ",
    ("Geoshield", "Hitek", "UVIRON", "SMP")
)
df = pd.DataFrame()
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
        if st.button("Process PDF", type="primary"):
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
elif option == 'Excel':
    uploaded_file = st.file_uploader(
        "Upload your Excel File", 
        type=["xlsx", "xls"]
    )

    if uploaded_file is not None:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        selected_sheet = st.selectbox(
            "Select Sheet",
            sheet_names
        )

        if st.button("Process Excel", type="primary"):
            with st.spinner("Processing Excel file..."):
                try:
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                    text = excel_to_text(df)
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

                except Exception as e:
                    st.error(f"Error processing Excel: {e}")

# IIF generation (only if we have matched rows)

if not df.empty:
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    @st.cache_data
    def load_meta(option_company):
        df = pd.read_excel("pages/CNT Data.xlsx", sheet_name="Sheet1", header=[0, 1])
        columns = list(df.columns[:9]) + [x for x in df.columns[9:] if option_company in x]
        column_names = [x[0] for x in df.columns[:9]] + [x[1] for x in df.columns[9:] if option_company in x]
                # flatten multi-level columns
        df_all = df[columns]
        df_all.columns = column_names
        return df_all.dropna(subset=['Description'])

    meta_df = load_meta(option_company)
    
    # Prepare CSV once
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = df.to_csv(index=False)

    matched_rows = []
    for _, row in df.iterrows():
        try:
            meta_match, factor = best_meta_match(row, meta_df, option_company)
            st.write(row["quantity"], row["amount"])
            # product, vlt, width, length, date, quantity, price, amount = row["product"], row["vlt"], row["width"], row["length"], row["date"], row["quantity"], row["price"], row["amount"]
            if meta_match is not None:
                type_code = meta_match["Type (Code)"]
                techpia_code = meta_match["Techpia (Code)"]
                description = meta_match["Description"]
                pi_unit_price = float(meta_match["Price"])
                po_unit_price = float(meta_match["PO Price"])
                slitting = meta_match["Width Slitting"]
                length = meta_match["Length"]
            matched_rows.append({
                "description": type_code,
                "product": row["description"],
                "vlt": row["vlt"],
                "quantity": row["quantity"],
                "price": row["price"],
                "amount": row["amount"]
            })
        except:
            st.write(row)
    df = pd.DataFrame(matched_rows)
    st.dataframe(df.head(10))

    with col1:
        download_button(
            df.to_csv(index=False),
            "output.csv",
            "Download CSV"
        )     
        
    from pages._utils import generate_purchase_order_iif, generate_sales_order_iif, load_qb_lists_from_iif, validate_items_against_qb

    @st.cache_data
    def load_qb_items():
        import pickle
        with open('pages/meta_iff.pkl', 'rb') as f:
            reference = pickle.load(f)
        items, vendors, customers = reference['items'], reference['vendors'], reference['customers']
        # items, vendors, customers = load_qb_lists_from_iif("pages/smp.IIF")
        return items, vendors, customers

    qb_items, qb_vendors, qb_customers = load_qb_items()

    matched_rows = json.loads(df.to_json(orient="records"))
    st.write(matched_rows[:2])
    
    missing_items = validate_items_against_qb(matched_rows, qb_items)
    
    if missing_items:
        st.write("Missing items: ", missing_items)
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
            f"purchase_order_{vendor_}.iif",
            "Download PO (.iif)"
        )
    
    with col3:
        download_button(
            st.session_state.so_iif,
            f"sales_order_{vendor_}.iif",
            "Download SO (.iif)"
        )
