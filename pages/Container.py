import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
import itertools
import json
from google import genai

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
- width: width in inches (integer)
- length: length in feet (integer)
- qty: quantity of rolls (integer)
- original_size_text: original size description exactly as shown in the source

VLT extraction rules:
1. If a VLT column exists, use it.
2. If VLT is embedded in the item name or film series, extract it:
   - Example: "i-Cool Carbon 02" → vlt = 2
   - Example: "i-Cool Premium IR 15" → vlt = 15
   - Remove the VLT number from the item name after extraction.
3. Do NOT guess VLT if it is not explicitly present.

Item / product rules:
- "Film Series" or similar column represents the item/product name.
- The item field should NOT include size, width, or VLT numbers.
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
- If a value cannot be confidently extracted, return "".

Table text begins below:

----
{data}
----
"""



# ================================================
# Load META file (meta.txt)
# ================================================
@st.cache_data
def load_meta():
    xlsx = pd.ExcelFile("pages/CNT Product Description (1).xlsx")
    dfs = {}
    
    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=[0, 1])
    
        # flatten multi-level columns
        df.columns = [
            f"{h1.strip()}_{h2.strip()}"
            for h1, h2 in df.columns
        ]
    
        # optional clean
        df.columns = (
            df.columns.str.replace(" ", "_")
                      .str.replace("(", "")
                      .str.replace(")", "")
        )
        st.write(sheet)
        st.dataframe(df.head(5))
        dfs[sheet] = df
    df_all = pd.concat(dfs.values(), ignore_index=True)
    # df = pd.read_csv("pages/meta.txt", sep="|")
    # df.columns = ["type_code", "techpia_code", "description", "unit_price"]
    # # Extract VLT value from Techpia code like “MEGAMAX 20”
    # df["vlt"] = df["techpia_code"].str.extract(r"(\d+)")
    # df["vlt"] = df["vlt"].astype(str)
    return df_all
meta_df = load_meta()

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
def consolidate_group(df):
    available = Counter()
    lengths = []

    for _, r in df.iterrows():
        qty = int(r["qty"])
        width = r["width"]
        length = r["length"]
        parts = r.get("parts")

        if length:
            lengths.append(length)

        if parts:
            # explode parts: each roll produces width-components
            for p in parts:
                available[p] += qty
        else:
            available[width] += qty

    final = consolidate_with_priority(available)

    # use majority length
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


def best_meta_match(row, meta_df):
    item = str(row["item"])
    vlt = int(str(row["vlt"]).strip().replace('%', ''))
    width_final = int(row["width"])
    # 1️⃣ Filter by matching VLT
    candidates = meta_df[meta_df["Proforma_Invoice_VLT"] == vlt]
    candidates["compare"] = candidates[["Proforma_Invoice_Description", "Proforma_Invoice_Width"]].apply(lambda x: str(x[0]) + ' ' + str(x[1]), axis=1)
    candidates = candidates[candidates["compare"].str.contains(str(width_final))]
    # st.dataframe(candidates.head(2))
    if candidates.empty:
        return None

    best_score = -1
    best_row = None

    for _, m in candidates.iterrows():
        meta_width = extract_width_from_meta(m["Purchase_Order_Description"])
        # 2️⃣ Width match (only when width_final < 60)
        if width_final < 60 and meta_width == width_final:
            width_score = 100
        else:
            width_score = 0

        # 3️⃣ Fuzzy match on item name
        # score1 = fuzz.token_set_ratio(item, m["description"])
        score1 = fuzz.token_set_ratio(str(row["composition"]), str(m["Proforma_Invoice_Width"]))
        score2 = fuzz.token_set_ratio(item, m["Proforma_Invoice_Description"])
        item_score = score1 + score2
        st.write(item, m, item_score)
        total_score = width_score + item_score

        if total_score > best_score:
            best_score = total_score
            best_row = m

    return best_row
    
# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Film Roll Width Consolidation (Simplified Version)")
client = genai.Client(api_key=st.secrets['gemini-api']['api_token'], http_options=http_options)    
models = client.models.list()
st.write('list of models')
st.write([x for x in models])
# option = "Proforma"
option_company = st.selectbox(
    "Company Name: ",
    ("GEOSHIELD", "HITEK", "Others")
)
option = st.selectbox(
    "Proforma vs Purchase Order",
    ("Proforma", "Purchase Order")
)
uploaded_files = st.file_uploader(
    "Upload one file to analyze (Excel, Image, PDF)",
    accept_multiple_files = True
)
with st.form("Proceed"):
    submitted = st.form_submit_button("Submit")

if submitted:
    all_rows = []
    if uploaded_files:
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
                        model="gemini-1.5-flash",
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
                model="gemini-1.5-flash",
                contents=[prompt]
            )
            json_text = out.text[out.text.find("["):out.text.rfind("]")+1]
            rows = json.loads(json_text)
    
            all_rows += rows
    
        df_norm = pd.DataFrame(all_rows)
        df_norm["item"] = df_norm["item"].ffill()
        st.write("Extracted information")
        st.dataframe(df_norm.head(30))
    
        # STEP 3: parse size fields properly
        widths = []
        lengths = []
        parts = []
    
        for t in df_norm["original_size_text"]:
            w, l, p = parse_size(t)
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
        st.write('consolidated')
        st.dataframe(df_final.head(20))
    
        # ============================================
        # 4. JOIN WITH META (by vlt)
        # ============================================
        matched_rows = []
    
        for _, r in df_final.iterrows():
            meta_match = best_meta_match(r, meta_df)
        
            if meta_match is not None:
                if option == "Proforma":
                    type_code = meta_match["Proforma_Invoice_Type_Code"]
                    techpia_code = meta_match["Purchase_Order_Techpia_Code"]
                    description = meta_match["Proforma_Invoice_Description"]
                    unit_price = float(meta_match["Proforma_Invoice_Unit_Price"])
                elif option == "Purchase Order":
                    type_code = meta_match["Purchase_Order_Type_Code"]
                    techpia_code = meta_match["Purchase_Order_Techpia_Code"]
                    description = meta_match["Purchase_Order_Description"]
                    unit_price = float(meta_match["Purchase_Order_Unit_Price"])
            else:
                type_code = ""
                techpia_code = ""
                description = ""
                unit_price = 0
        
            amount = unit_price * r["qty"]
        
            matched_rows.append({
                "type_code": type_code,
                "techpia_code": techpia_code,
                "description": description,
                "vlt": r["vlt"],
                "width": str(r["width"]) + ' (' + r["composition"] + ")" if '/' in r["composition"] else str(r["width"]),
                "length": r["length"],
                "thickness": "1.5" if 'IC-ALPU' not in type_code else "2.0",
                "quantity": r["qty"],
                "unit_price": f"${unit_price:,.2f}",
                "amount": f"${amount:,.2f}",
            })
        
        df_join = pd.DataFrame(matched_rows)

        df_join = df_join.sort_values("type_code")
        # Final column order
        # df_join = df_join[["techpia_code", "type_code", "description", "vlt", "width", "length", "thickness", "qty", "unit_price", "amount", "source_file", "composition", "item"]]
    
        st.subheader("Final Merged Table")
        st.dataframe(df_join, use_container_width=True)

st.download_button("Download CSV", df_join.to_csv(index=False), "output.csv")

