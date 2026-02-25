import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
import itertools
import json
from google import genai
import numpy as np

from google.genai import types 
retry_options = types.HttpRetryOptions(
    # The number of attempts to make before failing the request
    attempts=3, 
    # The initial delay (in seconds)
    initial_delay=1.0, 
    # The maximum delay (in seconds) between retries
    max_delay=60.0
)

# 3. Create the HttpOptions object
http_options = types.HttpOptions(
    retry_options=retry_options
)


# st.write(genai.__file__)
# ----------------------------
# Gemini Prompt
# ----------------------------
LLM_PROMPT = """
You are an expert in extracting and normalizing structured film inventory tables.

Given extracted vendor table text (from Excel, PDF, or image), return a JSON array.
Each JSON object represents ONE row with the following fields:

Required output fields:
- item: full product / film name WITHOUT VLT number
- series: product series or family name if identifiable (e.g., Carbon, Premium IR), else ""
- vlt: VLT percentage as an integer (e.g., 2, 5, 15). If not available, use ""
- width: total width in inches (integer)
- length: length in feet (integer)
- qty: quantity of rolls (integer)
- composition: list of integers representing an explicit width split (e.g., [36, 24]) IF AND ONLY IF explicitly provided in the source; otherwise null
- original_size_text: original size description exactly as shown in the source

Composition (combination) extraction rules:
1. If the source explicitly provides a width split, extract it in descreasing order (always):
   - Examples:
     - "60 (36/24)" → composition = [36, 24]
     - "36/24" → composition = [36, 24]
     - "24/36" → composition = [36, 24]
2. The sum of composition values MUST equal the total width.
3. If no explicit split is present, set composition = null.
4. Do NOT derive, infer, or compute composition values.

VLT extraction rules:
1. If a VLT column exists, use it.
2. If VLT is embedded in the item name or film series, extract it:
   - Example: "i-Cool Carbon 02" → vlt = 2
   - Example: "i-Cool Premium IR 15" → vlt = 15
   - Remove the VLT number from the item name after extraction.
3. Do NOT guess VLT if it is not explicitly present.

Item / product rules:
- "Film Series" or similar column represents the item/product name.
- The item field should NOT include size, width, length, or VLT numbers.
- Preserve original casing and wording as much as possible.

Size normalization rules:
- Normalize any of the following formats:
  - 40x100, 20x100
  - 36/24 inside parentheses
  - 60 (36/24)
  - 12, 20, 24, 36, 40 alone
  - X" x Y', X"XY', or similar formats
- Extract:
  - width → inches (integer)
  - length → feet (integer)
- If length is missing, infer from context ONLY if clearly specified (e.g., standard 100 ft roll). Otherwise leave blank.

Quantity rules:
- qty represents number of rolls.
- Extract numeric quantity only.

Strict rules:
- Return ONLY a valid JSON array.
- Do NOT include explanations, markdown, or comments.
- Do NOT invent data.
- If a value cannot be confidently extracted, return null or "" as appropriate.

Table text begins below:

----
{data}
----
"""


import extract_msg

def extract_text_from_msg(uploaded_file):
    msg = extract_msg.Message(uploaded_file)
    msg_message = msg.body or ""

    # Attachments?
    attachments = []
    for att in msg.attachments:
        attachments.append(att)
    return msg_message, attachments
    
# ----------------------------
# Width extraction
# ----------------------------
def parse_size(text):
    if not text:
        return None, None, None

    t = str(text)

    # "60 (36/24)" or "60 (24/12/12/12)"
    m = re.search(r'(?P<w>\d+)\s*\((?P<parts>[\d/\s,]+)\)', t)
    if m:
        width = int(m.group("w"))
        parts = [int(x) for x in re.split(r"[,/]", m.group("parts")) if x.strip().isdigit()]
        return width, None, parts

    # "40x100"
    m = re.search(r'(?P<w>\d+)\s*[xX]\s*(?P<l>\d+)', t)
    if m:
        return int(m.group("w")), int(m.group("l")), None

    # single width like 12, 20, 24 etc.
    m = re.search(r'(?P<w>\d+)', t)
    if m:
        return int(m.group("w")), None, None

    return None, None, None


# ================================================
# PRIORITY MERGE LOGIC (YOUR RULES)
# ================================================
PRIORITY_COMBOS = [
    [40, 20],
    [36, 24],
    [24, 12, 12, 12],
    [20, 20, 20],
    [12, 12, 12, 12, 12, 12],
]

def consolidate_with_priority(available: Counter):
    """
    available = Counter({width: qty})

    Returns list of:
        {composition, width_final, qty}
    """
    result = []

    # 1. Run priority combos first
    for combo in PRIORITY_COMBOS:
        need = Counter(combo)
        max_create = min(available[w] // need[w] for w in need if need[w] > 0)

        if max_create > 0:
            result.append({
                "composition": "/".join(str(x) for x in combo),
                "width_final": 60,
                "qty": max_create,
            })
            for w in need:
                available[w] -= need[w] * max_create

    # 2. fallback dynamic combos that sum to 60
    sizes = sorted([w for w, q in available.items() if q > 0 and w <= 60])

    # generate 2- and 3-part combos
    all_combos = []

    for a, b in itertools.combinations_with_replacement(sizes, 2):
        if a + b == 60:
            all_combos.append([a, b])

    for a, b, c in itertools.combinations_with_replacement(sizes, 3):
        if a + b + c == 60:
            all_combos.append([a, b, c])

    # sort larger-first
    all_combos.sort(key=lambda c: (-len(c), -max(c)))

    for combo in all_combos:
        need = Counter(combo)
        max_create = min(available[w] // need[w] for w in need)
        if max_create > 0:
            result.append({
                "composition": "/".join(map(str, combo)),
                "width_final": 60,
                "qty": max_create,
            })
            for w in need:
                available[w] -= need[w] * max_create

    # 3. leftovers
    leftovers = []
    for w, q in available.items():
        if q > 0:
            leftovers.append({
                "composition": str(w),
                "width_final": w,
                "qty": q
            })

    return result + leftovers


# ================================================
# CONSOLIDATE GROUPS BY ITEM + VLT
# ================================================
from collections import Counter
import pandas as pd

def consolidate_group(df):
    """
    Consolidate widths ONLY if composition is missing.
    If composition exists, return rows as-is (normalized shape).
    """

    # ---------- NEW: short-circuit if composition exists ----------
    if "composition" in df.columns:
        has_composition = df["composition"].notna().any()
    else:
        has_composition = False

    if has_composition:
        # Just normalize rows → no consolidation
        out = []
        for _, r in df.iterrows():
            if pd.isna(r["qty"]) or r["qty"] == "" or str(r["qty"]) == "NaN":
                r["qty"] = 0
            out.append({
                "width_final": r["width"],
                "composition": "/".join([str(x) for x in r["composition"]]) if r["composition"] is not None else "",
                "qty": int(r["qty"]) if r["qty"] is not None and r["qty"] != "" and str(r["qty"]) != "NaN" else 0,
                "length": r["length"]
            })
        return out
    # --------------------------------------------------------------

    # ---------- Existing logic (unchanged) ----------
    available = Counter()
    lengths = []

    for _, r in df.iterrows():
        try:
            qty = int(r["qty"])
        except:
            qty = 0
        width = r["width"]
        length = r["length"]
        parts = r.get("parts")  # backward compatibility

        if length:
            lengths.append(length)

        if parts:
            for p in parts:
                available[p] += qty
        else:
            available[width] += qty

    final = consolidate_with_priority(available)

    length_val = max(set(lengths), key=lengths.count) if lengths else None

    for r in final:
        r["length"] = length_val

    return final


from rapidfuzz import fuzz

def extract_width_from_meta(desc):
    # Example: 'Megamax 20% 40"X100' → 40
    m = re.search(r'(\d+)\s*["]?X', desc)
    if m:
        return int(m.group(1))
    return None


def best_meta_match(row, meta_df, option_company):
    st.write("Here processing: ", row)
    item = str(row["item"])
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
    if width_final == 12:
        row["composition"] = "5*12"
        width_final = 60
        factor = 5

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

    for _, m in candidates.iterrows():
        if str(row["composition"]) != 'nan' and '/' not in str(row["composition"]) and '*' not in str(row["composition"]):
            row["composition"] = 'nan'
        st.write("description value: ", m["QB Description"], m["Description"], item, any([x.lower() in m["QB Description"].lower() for x in item.split()]))
        if any([x.lower() in m["QB Description"].lower() for x in item.split()]) or any([x.lower() in m["Description"].lower() for x in item.split()]):
            total_score = -1
            multiplier = sum([[0,1][x.lower() in m["QB Description"].lower() or x.lower() in m["Description"]] for x in item.split()])
            # if factor != 1:
            #     st.write('special case: ', m["Width"], row["composition"], ('/' in str(m["Width"]) or '*' in str(m["Width"])), str(row["composition"]) != 'nan', (str(row["composition"]) != 'nan' and ('/' in str(m["Width"]) or '*' in str(m["Width"]))))
            if (str(row["composition"]) == 'nan' and '/' not in str(m["Width"])) or (str(row["composition"]) != 'nan' and ('/' in str(m["composition"]) or '*' in str(m["composition"]))):
                meta_width = extract_width_from_meta(m["Description"])
                st.write('meta_width: ', meta_width)
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
                st.write(width_score, item_score, total_score)
            if total_score > best_score:
                # st.write(item, m["Width"], m["Description"], total_score)
                best_score = total_score
                best_row = m

    return best_row, factor
    
# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Film Roll Width Consolidation (Simplified Version)")
client = genai.Client(api_key=st.secrets['gemini-api']['api_token'], http_options=http_options)    
models = client.models.list()
# option = "Proforma"
option_company = st.selectbox(
    "Company Name: ",
    ("Geoshield", "Hitek", "UVIRON", "SMP")
)
# option = st.selectbox(
#     "Proforma vs Purchase Order",
#     ("Proforma", "Purchase Order")
# )
uploaded_files = st.file_uploader(
    "Upload one file to analyze (Excel, Image, PDF)",
    accept_multiple_files = True
)
with st.form("Proceed"):
    submitted = st.form_submit_button("Submit")

df_join = pd.DataFrame()
if submitted:
    # ================================================
    # Load META file (meta.txt)
    # ================================================
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
    st.write("Loading metadata completed...")
    
    all_rows = []
    if uploaded_files:
        st.write("Reading the loaded files")
        for uploaded in uploaded_files:
            suffix = uploaded.name.lower()
        
            # STEP 1: extract data from the input file
            if suffix.endswith(("xlsx", "xls")):
                df = pd.read_excel(uploaded)
                raw_data = df.to_csv(index=False)
            elif suffix.endswith(".msg"):
                text, attachments = extract_text_from_msg(uploaded)
            
                raw_data = text
            else:
                # Image or PDF → use Gemini Vision
                from PIL import Image
                import fitz
                if suffix.endswith(("png", "jpg", "jpeg")):
                    img = Image.open(uploaded)
                    result = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[img]
                    )
                    
                    raw_data = result.text
                elif suffix.endswith("pdf"):
                    pdf = fitz.open(stream=uploaded.read(), filetype="pdf")
                    text = ""
                    for p in pdf:
                        text += p.get_text()
                    raw_data = text
                else:
                    st.error("Unsupported file")
        
            # STEP 2: Normalize table using LLM
            prompt = LLM_PROMPT.format(data=raw_data)
            out = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt]
            )
            json_text = out.text[out.text.find("["):out.text.rfind("]")+1]
            rows = json.loads(json_text)
    
            all_rows += rows
    
        df_norm = pd.DataFrame(all_rows)
        df_norm["item"] = df_norm["item"].ffill()
        df_norm['width'] = df_norm['width'].fillna(method='ffill')
        df_norm['width'] = df_norm['width'].replace(r"^\s*$", np.nan, regex=True).ffill()
        df_norm['original_size_text'] = df_norm['original_size_text'].fillna(method='ffill')
        df_norm['original_size_text'] = df_norm['original_size_text'].replace(r"^\s*$", np.nan, regex=True).ffill()
        st.write("Extracted information")
        st.dataframe(df_norm.head(200))

        # STEP 3: parse size fields properly
        widths = []
        lengths = []
        parts = []
    
        for ww, t in zip(df_norm["width"], df_norm["original_size_text"]):
            w, l, p = parse_size(t)
            if w is None:
                widths.append(ww)
            else:
                widths.append(w)
            lengths.append(l)
            parts.append(p)
    
        df_norm["width"] = widths
        df_norm["length"] = lengths
        df_norm["parts"] = parts
    
        # STEP 4: consolidate width sum to 60
        final_rows = []
    
        for (item, vlt), group in df_norm.groupby(["item", "vlt"]):
            out = consolidate_group(group)
            for r in out:
                final_rows.append({
                    "item": item,
                    "vlt": vlt,
                    "composition": r["composition"],
                    "width": r["width_final"],
                    "length": r["length"],
                    "qty": r["qty"]
                })
    
        df_final = pd.DataFrame(final_rows)
        # only process row with quantity > 0
        df_final = df_final[df_final['qty'] > 0]
        st.write('consolidated')
        st.dataframe(df_final.head(200))
    
        # ============================================
        # 4. JOIN WITH META (by vlt)
        # ============================================
        matched_rows = []
    
        for _, r in df_final.iterrows():
            meta_match, factor = best_meta_match(r, meta_df, option_company)
        
            if meta_match is not None:
                type_code = meta_match["Type (Code)"]
                techpia_code = meta_match["Techpia (Code)"]
                description = meta_match["Description"]
                pi_unit_price = float(meta_match["Price"])
                po_unit_price = float(meta_match["PO Price"])
                length = meta_match["Length"]
            else:
                type_code = ""
                techpia_code = ""
                description = ""
                pi_unit_price = 0
                po_unit_price = 0
                length = None

            if factor != 1:
                r["qty"] = int(r["qty"] / factor)
                
            pi_amount = pi_unit_price * r["qty"]
            po_amount = po_unit_price * r["qty"]
        
            matched_rows.append({
                "type_code": type_code,
                "techpia_code": techpia_code,
                "description": description,
                "vlt": r["vlt"],
                "width": str(r["width"]) + ' (' + r["composition"] + ")" if '/' in r["composition"] else str(r["width"]),
                "length": r["length"] if r["length"] is not None and r["length"] != "" and r["length"] != 0 else length,
                "thickness": "1.5" if 'IC-ALPU' not in type_code else "2.0",
                "quantity": r["qty"],
                "pi_unit_price": f"{pi_unit_price:,.2f}",
                "po_unit_price": f"{po_unit_price:,.2f}",
                "pi_amount": f"{pi_amount:,.2f}",
                "po_amount": f"{po_amount:,.2f}",
            })
        
        df_join = pd.DataFrame(matched_rows)

        df_join = df_join.sort_values("type_code")
        # Final column order
        # df_join = df_join[["techpia_code", "type_code", "description", "vlt", "width", "length", "thickness", "qty", "unit_price", "amount", "source_file", "composition", "item"]]
    
        st.subheader("Final Merged Table")
        st.dataframe(df_join, use_container_width=True)

        # Download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button("Download CSV", df_join.to_csv(index=False), "output.csv", "text/csv")

        # IIF generation (only if we have matched rows)
        if not df_join.empty:
            from pages._utils import generate_purchase_order_iif, generate_sales_order_iif, load_qb_lists_from_iif, validate_items_against_qb
            
            # Load QB items for validation
            @st.cache_data
            def load_qb_items():
                items, vendors, customers = load_qb_lists_from_iif("pages/smp.IIF")
                return items, vendors, customers
            
            qb_items, qb_vendors, qb_customers = load_qb_items()
            
            # Validate items
            missing_items = validate_items_against_qb(matched_rows, qb_items)
            if missing_items:
                st.warning(f"⚠️ Warning: {len(missing_items)} item(s) not found in QuickBooks:\n" + 
                           "\n".join(missing_items[:5]) + 
                           (f"\n... and {len(missing_items) - 5} more" if len(missing_items) > 5 else ""))
            else:
                st.success("✅ All items validated against QuickBooks.")
            
            # Map company to vendor name (you may need to adjust these)
            vendor_map = {
                "Geoshield": "Geoshield",
                "Hitek": "Hitek",
                "UVIRON": "UVIRON",
                "SMP": "SMP"
            }
            vendor_name = vendor_map.get(option_company, option_company)
            
            # Generate PO IIF
            po_iif_content = generate_purchase_order_iif(matched_rows, vendor_name=vendor_name)
            
            with col2:
                st.download_button(
                    "Download PO (.iif)", 
                    po_iif_content, 
                    f"purchase_order_{option_company}.iif", 
                    "text/plain"
                )
            
            # Generate SO IIF (using a default customer name - can be customized)
            so_iif_content = generate_sales_order_iif(matched_rows, customer_name="Default Customer")
            
            with col3:
                st.download_button(
                    "Download SO (.iif)", 
                    so_iif_content, 
                    f"sales_order_{option_company}.iif", 
                    "text/plain"
                )

