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
    df = pd.read_csv("meta.txt", sep="|")
    df.columns = ["type_code", "techpia_code", "description", "unit_price"]
    # Extract VLT value from Techpia code like “MEGAMAX 20”
    df["vlt"] = df["techpia_code"].str.extract(r"(\d+)")
    df["vlt"] = df["vlt"].astype(str)
    return df
    
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


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Film Roll Width Consolidation (Simplified Version)")
client = genai.Client(api_key=st.secrets['gemini-api']['api_token'])    
# models = client.models.list()
# st.write([m.name for m in models])
uploaded_files = st.file_uploader(
    "Upload one or multiple files (Excel, Image, PDF)",
)

if uploaded:
    suffix = uploaded.name.lower()

    # STEP 1: extract data from the input file
    if suffix.endswith(("xlsx", "xls")):
        df = pd.read_excel(uploaded)
        raw_data = df.to_csv(index=False)
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

    df_norm = pd.DataFrame(rows)
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
    df_join = df_final.merge(meta_df, on="vlt", how="left")

    # Thickness is not provided; set empty
    df_join["thickness"] = ""

    # Amount = Unit Price × Qty
    df_join["amount"] = df_join["unit_price"].fillna(0) * df_join["qty"]

    # Final column order
    df_join = df_join[
        [
            "techpia_code",
            "type_code",
            "description",
            "vlt",
            "width",
            "length",
            "thickness",
            "qty",
            "unit_price",
            "amount",
            "source_file",
            "composition",
            "item"
        ]
    ]

    st.subheader("Final Merged Table")
    st.dataframe(df_join, use_container_width=True)

    st.download_button("Download CSV", df_join.to_csv(index=False), "output.csv")

