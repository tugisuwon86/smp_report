import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
import itertools
import json
from google import genai
st.write(genai.__file__)
# ----------------------------
# Gemini Prompt
# ----------------------------
LLM_PROMPT = """
You are an expert in extracting structured film inventory tables.

Given extracted vendor table text, return a JSON array with rows containing:
- item: the product name or film type
- vlt: VLT percentage (extract if available, else "")
- width: width in inches (integer)
- length: length in feet (integer)
- qty: quantity of rolls (integer)
- original_size_text: original size description

Normalize any size formats such as:
- 40x100
- 20x100
- 36/24 inside parentheses
- 60 (36/24)
- 12, 20, 24, 36, 40 alone
- X"XY' formats  
Extract numeric values reliably.

Return ONLY a JSON array.
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
    st.write(item, vlt, width_final)
    # 1️⃣ Filter by matching VLT
    candidates = meta_df[meta_df["Proforma_Invoice_VLT"] == vlt]
    st.dataframe(candidates.head(2))
    candidates["compare"] = candidates["Proforma_Invoice_Description"].str + " " + candidates["Proforma_Invoice_Width"].str
    candidates = candidates[candidates["compare"].str.contains(str(width_final))]
    if candidates.empty:
        return None

    best_score = -1
    best_row = None

    for _, m in candidates.iterrows():
        meta_width = extract_width_from_meta(m["Proforma_Invoice_Width"])
        st.write('candidate: ', meta_width, width_final)
        # 2️⃣ Width match (only when width_final < 60)
        if width_final < 60 and meta_width == width_final:
            width_score = 100
        else:
            width_score = 0

        # 3️⃣ Fuzzy match on item name
        score1 = fuzz.token_set_ratio(item, m["description"])
        score2 = fuzz.token_set_ratio(item, m["techpia_code"])
        item_score = max(score1, score2)

        total_score = width_score + item_score

        if total_score > best_score:
            best_score = total_score
            best_row = m

    return best_row
    
# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Film Roll Width Consolidation (Simplified Version)")
client = genai.Client(api_key=st.secrets['gemini-api']['api_token'])    
# models = client.models.list()
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

    df_norm = pd.DataFrame(rows)
    df_norm["item"] = df_norm["item"].ffill()
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

    # ============================================
    # 4. JOIN WITH META (by vlt)
    # ============================================
    matched_rows = []

    for _, r in df_final.iterrows():
        meta_match = best_meta_match(r, meta_df)
    
        if meta_match is not None:
            if option == "Proforma":
                type_code = meta_match["Proforma_Invoice_Type_Code"]
                techpia_code = meta_match["techpia_code"]
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
            "width": str(r["width"]) + ' (' + r["composition"] + ")" if '/' in r["composition"] else "",
            "length": r["length"],
            "thickness": "1.5",
            "quantity": r["qty"],
            "unit_price": f"${unit_price:,.2f}",
            "amount": f"${amount:,.2f}",
        })
    
    df_join = pd.DataFrame(matched_rows)

    # Final column order
    # df_join = df_join[["techpia_code", "type_code", "description", "vlt", "width", "length", "thickness", "qty", "unit_price", "amount", "source_file", "composition", "item"]]

    st.subheader("Final Merged Table")
    st.dataframe(df_join, use_container_width=True)

    st.download_button("Download CSV", df_join.to_csv(index=False), "output.csv")

