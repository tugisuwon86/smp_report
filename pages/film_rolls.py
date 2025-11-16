import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
import itertools
import json
from google import genai

# ----------------------------
# Gemini Prompt
# ----------------------------
LLM_PROMPT = """
You are an expert data extraction agent specializing in window tint film inventory.

Normalize the table to JSON rows with the following fields:
- item
- vlt
- width (numeric, inches)
- length (numeric, feet)
- qty (integer)
- original_size_text

Ensure no invented values. Return ONLY a JSON array.
Data:
{data}
"""

# ----------------------------
# Width extraction
# ----------------------------
def parse_size(text):
    if text is None:
        return None, None, None

    t = str(text)

    # Formats like 60 (36/24)
    m = re.search(r'(?P<w>\d{2,3})\s*\((?P<parts>[\d\s\/,]+)\)', t)
    if m:
        width = int(m.group("w"))
        parts = [int(x) for x in re.split(r'[,/]', m.group("parts")) if x.strip().isdigit()]
        return width, None, parts

    # Formats like 40x100
    m = re.search(r'(?P<w>\d+)\s*[xX]\s*(?P<l>\d+)', t)
    if m:
        return int(m.group("w")), int(m.group("l")), None

    # Single width like 12, 20, 36, 40
    m = re.search(r'(?P<w>\d{1,3})', t)
    if m:
        return int(m.group("w")), None, None

    return None, None, None


# ----------------------------
# Width consolidation
# ----------------------------
def consolidate_group(df):
    """
    Input: df rows for same item + vlt
    Output: list of dicts with composition, width_final, length, qty
    """

    available = Counter()
    existing_60 = 0
    lengths = []

    # Build counts
    for _, r in df.iterrows():
        qty = int(r["qty"])
        length = r["length"]
        if length:
            lengths.append(length)

        width = r["width"]
        parts = r.get("parts")

        if width == 60 and not parts:
            existing_60 += qty
            continue

        if parts:
            # parts sum 60 but user may want recombination → treat parts individually
            for p in parts:
                available[p] += qty
        else:
            available[width] += qty

    # Combine widths to sum to 60
    results = []

    def combos_for_target(cnt: Counter, target=60):
        sizes = sorted(cnt.keys())
        combos = []
        for a, b in itertools.combinations_with_replacement(sizes, 2):
            if a + b == target:
                combos.append([a, b])
        for a, b, c in itertools.combinations_with_replacement(sizes, 3):
            if a + b + c == target:
                combos.append([a, b, c])
        combos.sort(key=lambda x: (-len(x), -max(x)))
        return combos

    while True:
        combos = combos_for_target(available)
        if not combos:
            break
        c = combos[0]   # best combo
        need = Counter(c)
        max_batch = min(available[w] // need[w] for w in need.keys())

        if max_batch == 0:
            break

        results.append({
            "composition": "/".join(str(i) for i in c),
            "width_final": 60,
            "qty": max_batch
        })

        for w in need:
            available[w] -= need[w] * max_batch

    # Add standalone leftover widths
    for w, q in available.items():
        if q > 0:
            results.append({
                "composition": str(w),
                "width_final": w,
                "qty": q
            })

    # Add existing 60
    if existing_60 > 0:
        results.insert(0, {
            "composition": "60",
            "width_final": 60,
            "qty": existing_60
        })

    # Use most common length
    length_val = max(set(lengths), key=lengths.count) if lengths else None

    final = []
    for r in results:
        final.append({
            "composition": r["composition"],
            "width": r["width_final"],
            "length": length_val,
            "qty": r["qty"]
        })

    return final


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Film Roll Width Consolidation (Simplified Version)")

if "vision" not in st.session_state:
    client = genai.Client(api_key=st.secrets['gemini-api']['api_token'])
    st.session_state["vision"] = client
    
uploaded = st.file_uploader("Upload Excel / Image / PDF")

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
            result = st.session_state["vision"].models.generate_content(
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
    llm = st.session_state["llm"]
    prompt = LLM_PROMPT.format(data=raw_data)
    out = llm.generate_text(prompt)
    json_text = out.text[out.text.find("["):out.text.rfind("]")+1]
    rows = json.loads(json_text)

    df_norm = pd.DataFrame(rows)

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
                "width": r["width"],
                "length": r["length"],
                "qty": r["qty"]
            })

    df_final = pd.DataFrame(final_rows)

    st.success("Final Consolidated Table")
    st.dataframe(df_final)

    st.download_button("Download CSV", df_final.to_csv(index=False), "output.csv")

