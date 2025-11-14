import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
import itertools

st.set_page_config(page_title="Roll Width Consolidation", layout="wide")


#######################
# --- Helper utils ---
#######################

SIZE_PATTERNS = [
    # examples: 40x100, 40 x 100, 40"X100', 60 (36/24)
    r'(?P<w1>\d{1,3})\s*[xX×]\s*(?P<l1>\d{1,3})',            # 40x100
    r'(?P<w2>\d{1,3})["”]?\s*[xX]\s*(?P<l2>\d{1,3})',        # 40"X100
    r'(?P<final>\d{1,3})\s*\(\s*(?P<parts>[\d/,\s]+)\s*\)',  # 60 (36/24) or 60 (24/12/12)
    r'(?P<w_only>\d{1,3})\s*(?:inch|in|")\b',                # width only "40 inch"
]


def extract_width_length_from_text_cell(cell_text: str):
    """
    Try to parse widths and length from a single text cell.
    Returns: (width_primary:int or None, length:int or None, composition:str or None, parts:list[int])
    """
    if not isinstance(cell_text, str):
        cell_text = str(cell_text)

    txt = cell_text.strip()
    # look for final w (with parts)
    m = re.search(r'(?P<final>\d{1,3})\s*\(\s*(?P<parts>[\d/,\s]+)\s*\)', txt)
    if m:
        final = int(m.group('final'))
        parts_raw = re.split(r'[/,]', m.group('parts'))
        parts = [int(p.strip()) for p in parts_raw if p.strip().isdigit()]
        return final, None, "/".join(str(p) for p in parts), parts

    # look for WxL
    m = re.search(r'(?P<w>\d{1,3})\s*[xX×]\s*(?P<l>\d{1,3})', txt)
    if m:
        w = int(m.group('w')); l = int(m.group('l'))
        return w, l, None, [w]

    # single width mention "40" or "60"
    m = re.search(r'(?<!\d)(?P<w>\d{1,3})(?:\s*(?:inch|in|\"|”))?(?!\d)', txt)
    if m:
        w = int(m.group('w'))
        return w, None, None, [w]

    return None, None, None, []


def normalize_columns(df: pd.DataFrame):
    """
    Attempt to find or map the key columns to standard names:
    item, vlt, width, length, qty, description
    """
    colmap = {}
    lower_to_col = {c.lower(): c for c in df.columns}

    # heuristics
    def find_candidate(keywords):
        for k in keywords:
            if k in lower_to_col:
                return lower_to_col[k]
        # fuzzy: any column containing keyword
        for c in df.columns:
            lc = c.lower()
            for k in keywords:
                if k in lc:
                    return c
        return None

    colmap['item'] = find_candidate(['item', 'product', 'techpia', 'description', 'desc', 'code', 'type', 'name'])
    colmap['vlt'] = find_candidate(['vlt', 'visible', 'vlt%','vl%','vl'])
    colmap['width'] = find_candidate(['width', 'size', 'size"', 'size x', 'size x length', 'size (inch)', 'width(inch)', 'size'])
    colmap['length'] = find_candidate(['length', 'len', "ft", "feet"])
    colmap['qty'] = find_candidate(['qty', 'quantity', 'roll', 'rolls', 'amount'])
    colmap['unit_price'] = find_candidate(['unit price','unit_price','price'])
    colmap['description'] = find_candidate(['description','desc','note','notes'])

    # fallback - if width isn't explicitly found but there's a column with strings like '40x100' use it
    if colmap['width'] is None:
        for c in df.columns:
            sample = " ".join(df[c].astype(str).sample(min(5, len(df))).tolist()).lower()
            if re.search(r'\d{1,3}\s*[xX×]\s*\d{1,3}', sample) or re.search(r'\d{1,3}\s*\(\s*[\d/,\s]+\s*\)', sample):
                colmap['width'] = c
                break

    # If any mapping is None, leave as None — we'll handle missing
    return colmap


def build_standardized_rows(df: pd.DataFrame):
    """
    Build a canonical dataframe with columns:
      item, vlt, width (primary), length, qty, composition (if given), parts (list)
    """
    rows = []
    colmap = normalize_columns(df)
    for _, r in df.iterrows():
        item = r[colmap['item']] if colmap['item'] and colmap['item'] in r else None
        vlt = r[colmap['vlt']] if colmap['vlt'] and colmap['vlt'] in r else None
        qty = r[colmap['qty']] if colmap['qty'] and colmap['qty'] in r else None
        description = r[colmap['description']] if colmap['description'] and colmap['description'] in r else None
        width_cell = r[colmap['width']] if colmap['width'] and colmap['width'] in r else ''
        length_cell = r[colmap['length']] if colmap['length'] and colmap['length'] in r else ''

        # attempt to parse width and length from width cell and length cell
        w, l, composition, parts = extract_width_length_from_text_cell(str(width_cell))
        # if length empty try parse from length cell
        if not l:
            _, l2, _, _ = extract_width_length_from_text_cell(str(length_cell))
            if l2:
                l = l2

        # if composition empty but width_cell contains parentheses parts (like '60 (36/24)')
        if not composition:
            m = re.search(r'\(\s*([\d/,\s]+)\s*\)', str(width_cell))
            if m:
                parts_raw = re.split(r'[,/]', m.group(1))
                parts = [int(p.strip()) for p in parts_raw if p.strip().isdigit()]
                composition = "/".join(str(p) for p in parts)

        # qty normalization
        try:
            qty_n = int(float(str(qty).replace(',',''))) if qty not in [None, '', 'nan', 'NaN'] else None
        except:
            qty_n = None

        rows.append({
            'item': str(item) if item is not None else '',
            'vlt': str(vlt) if vlt is not None else '',
            'width_raw': str(width_cell),
            'width': int(w) if w else None,
            'length': int(l) if l else None,
            'composition': composition if composition else None,
            'parts': parts if parts else [],
            'qty': qty_n if qty_n else 0,
            'description': str(description) if description else ''
        })
    return pd.DataFrame(rows)


###############################
# --- Consolidation routine ---
###############################

def find_feasible_combo(available_counts: Counter):
    """
    Find any feasible combo of sizes (length 2 or 3) whose sum == 60.
    Preference ordering: combos with largest max part, then by fewer parts (2 before 3).
    Returns tuple(parts_list) or None
    """
    sizes = sorted([s for s, c in available_counts.items() if c > 0])
    # generate unique combos of length 2 or 3 (order doesn't matter)
    combos = set()
    # pairs
    for a, b in itertools.combinations_with_replacement(sizes, 2):
        if a + b == 60:
            combos.add(tuple(sorted((a, b), reverse=True)))
    # triplets
    for a, b, c in itertools.combinations_with_replacement(sizes, 3):
        if a + b + c == 60:
            combos.add(tuple(sorted((a, b, c), reverse=True)))

    if not combos:
        return None

    # sort combos by priority: larger first element, then second, then prefer shorter combos (2 before 3)
    combos = sorted(list(combos), key=lambda t: (t + (0,))[:3], reverse=True)
    combos = sorted(combos, key=lambda t: (len(t),), reverse=False)  # prefer smaller length first
    # combine sortings: first prefer length 2, then by descending parts
    combos = sorted(list(combos), key=lambda t: (len(t), -t[0], - (t[1] if len(t)>1 else 0), - (t[2] if len(t)>2 else 0)))
    # find first feasible given counts
    for combo in combos:
        ok = True
        tmp = Counter()
        for part in combo:
            tmp[part] += 1
        for p, need in tmp.items():
            if available_counts[p] < need:
                ok = False; break
        if ok:
            return list(combo)
    return None


def consolidate_group(rows_sub: pd.DataFrame):
    """
    rows_sub: DataFrame for a single (item,vlt,length) group.
    returns list of consolidated output rows (dicts)
    """
    # accumulate counts per width (include widths from composition parts if provided)
    width_counts = Counter()
    # also keep any existing 60" rolls and treat them as final
    existing_60 = 0
    records = []

    for _, r in rows_sub.iterrows():
        qty = int(r['qty']) if r['qty'] else 0
        if qty <= 0:
            continue

        # if parts exist and composition is e.g. [36,24] and width == 60, we treat as existing 60
        if r['width'] == 60 or (r['composition'] and r['width'] and r['width'] >= 60):
            existing_60 += qty
            continue

        # prefer to use explicit parts if present and sum parts == width
        parts = r['parts'] if r['parts'] else [r['width']] if r['width'] else []
        # If parts length 1 just the width
        # For simplicity, assign the qty to the primary width value
        if parts:
            for p in parts:
                width_counts[p] += qty  # note: this duplicates qty to each part if parts>1; but typical rows either have single width or are 60 with parts.
                # To avoid duplication when parts reflect decomposition of one roll into multiple widths,
                # we only add to primary width if single part. So adjust:
            # Better approach: if multiple parts but row width==60, we should treat as existing_60 (handled above)
            # So here we adjust: if parts>1 and width not 60, ignore parts and use width
            if len(parts) > 1 and r['width'] and r['width'] != 60:
                # fallback to primary width
                width_counts[r['width']] += qty
            # if single part, we already added
        else:
            # unknown parts - try width
            if r['width']:
                width_counts[r['width']] += qty

    # Now create consolidated 60" rolls greedily
    created_60_rows = []
    available = Counter(width_counts)  # copy

    # First, remove widths that are already >60 or unsupported (we'll treat them as standalone later)
    # We aim to combine widths that are <=60
    for w in list(available):
        if w > 60:
            # leave them alone; will be output later as standalone
            continue

    while True:
        combo = find_feasible_combo(available)
        if not combo:
            break
        # create one combined roll per available set (take 1 of each)
        # But we want to create as many as possible of this combo: compute min multiplicity
        counts_needed = Counter(combo)
        max_create = min(available[p] // counts_needed[p] for p in counts_needed)
        # create batch
        composition_str = "/".join(map(str, combo))
        created_60_rows.append({
            'composition': composition_str,
            'width_final': 60,
            'qty': max_create
        })
        # decrement
        for p in counts_needed:
            available[p] -= counts_needed[p] * max_create

    # After combos created, remaining widths become standalone rows
    leftovers = []
    for w, c in sorted(available.items(), reverse=True):
        if c > 0:
            leftovers.append({'composition': str(w), 'width_final': w, 'qty': c})

    # include any existing 60" rolls
    if existing_60 > 0:
        created_60_rows.insert(0, {'composition': '60', 'width_final': 60, 'qty': existing_60})

    # final output rows annotated with item/vlt/length
    out = []
    for r in created_60_rows + leftovers:
        out.append({
            'composition': r['composition'],
            'width': r['width_final'],
            'length': int(rows_sub['length'].dropna().iloc[0]) if rows_sub['length'].notna().any() else None,
            'qty': int(r['qty'])
        })
    return out


##########################
# --- Streamlit UI ------
##########################

def app():
    st.title("Film Roll Width Consolidation")
    st.markdown("""
    Upload an extracted table (Excel/CSV) or paste your extracted text/JSON.  
    The tool will attempt to parse sizes (e.g. `40x100`, `60 (36/24)`, `40`) and quantities, then consolidate widths into 60\" rolls where possible.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload Excel / CSV / Image / PDF / MSG (optional)", type=['xlsx','xls','csv','pdf','png','jpg','jpeg','msg'])
        pasted = st.text_area("Or paste extracted table (CSV, JSON array, or plain text)", height=180)
    with col2:
        st.info("Workflow:\n1) Upload or paste\n2) Inspect parsed table\n3) Press 'Consolidate'")

        st.write("Options:")
        use_llm_parse = st.checkbox("Use LLM for parsing (optional)", value=False)
        # (LLM integration placeholder - user likely already has Gemini in their app)
        if use_llm_parse:
            st.caption("LLM parsing is optional. This app contains a heuristic parser; enable LLM if you want a language-model-powered parse. Add your integration separately.")

    df_input = None

    # Attempt to load uploaded file into DataFrame if possible
    if uploaded:
        name = uploaded.name.lower()
        try:
            if name.endswith(('xlsx','xls')):
                df_input = pd.read_excel(uploaded, engine='openpyxl' if name.endswith('xlsx') else None)
            elif name.endswith('csv'):
                df_input = pd.read_csv(uploaded)
            elif name.endswith('pdf'):
                # best-effort: try to extract text pages (requires fitz/pymupdf)
                try:
                    import fitz
                    uploaded_bytes = uploaded.read()
                    pdf = fitz.open(stream=uploaded_bytes, filetype='pdf')
                    texts = []
                    for p in pdf:
                        texts.append(p.get_text("text"))
                    pasted = "\n".join(texts)
                    st.success("Extracted text from PDF into the paste box — please verify and press Parse.")
                except Exception as e:
                    st.warning("PDF reading not available (fitz missing) or failed. Please paste extracted text or upload an Excel/CSV instead.")
                    st.exception(e)
            elif name.endswith(('png','jpg','jpeg')):
                # if you have OCR, you probably run it upstream and paste output. Here we show a placeholder.
                st.info("Image uploaded. This page expects extracted text. If you have OCR upstream, paste its output in the text box.")
            elif name.endswith('.msg'):
                try:
                    import extract_msg
                    msg = extract_msg.Message(uploaded)
                    body = msg.body
                    pasted = (pasted + "\n\n" + body) if pasted else body
                    st.success("Extracted email body into paste box. Verify and press Parse.")
                except Exception as e:
                    st.warning("extract_msg not installed or failed. Please paste message body manually.")
        except Exception as e:
            st.error("Failed to read uploaded file into a table; please paste table text or upload an Excel/CSV.")
            st.exception(e)

    # If user pasted something that looks like JSON array or CSV, try parse
    if pasted and df_input is None:
        txt = pasted.strip()
        # JSON array?
        if txt.startswith('['):
            try:
                df_input = pd.read_json(txt)
            except Exception:
                pass
        # CSV table?
        if df_input is None:
            try:
                df_input = pd.read_csv(BytesIO(txt.encode()))
            except Exception:
                # fallback: try to parse simple whitespace-separated table into columns
                lines = [l for l in txt.splitlines() if l.strip()]
                if len(lines) > 1 and any(re.search(r'\d', lines[0])):
                    # naive parse using split by multiple spaces or tabs
                    parsed = [re.split(r'\s{2,}|\t', ln.strip()) for ln in lines]
                    maxcols = max(len(p) for p in parsed)
                    df_input = pd.DataFrame(parsed)
                    st.caption("Parsed pasted text heuristically into a table — verify columns.")
                else:
                    # cannot parse into dataframe; let users press parse to build rows from text
                    pass

    if df_input is not None:
        st.subheader("Parsed Input Table (detected)")
        st.dataframe(df_input.head(200))
        st.markdown("### Standardized rows (attempting to extract width/length/qty)")
        canonical = build_standardized_rows(df_input)
        st.dataframe(canonical.head(500))

        if st.button("Consolidate widths to 60\""):
            # group by item,vlt,length (or item,vlt if length missing)
            group_cols = []
            if 'item' in canonical.columns:
                group_cols.append('item')
            if 'vlt' in canonical.columns:
                group_cols.append('vlt')
            # ensure grouping keys present as columns (they are)
            group_cols = ['item', 'vlt', 'length']

            results = []
            summary = {
                'total_original_rolls': int(canonical['qty'].sum()),
                'total_after_consolidation': 0,
                'widths_60_inch_created': 0,
                'leftover_rolls': {}
            }

            # fill any NaN length with a placeholder None
            canonical['length'] = canonical['length'].where(pd.notnull(canonical['length']), None)

            grouped = canonical.groupby(['item', 'vlt', 'length'], dropna=False)
            for key, sub in grouped:
                item, vlt, length = key
                out_rows = consolidate_group(sub)
                for o in out_rows:
                    results.append({
                        'item': item,
                        'vlt': vlt,
                        'composition': o['composition'],
                        'width_final': o['width'],
                        'length': o['length'] if o['length'] is not None else length,
                        'qty': o['qty'],
                    })

            df_out = pd.DataFrame(results)
            if df_out.empty:
                st.warning("No rows produced - check source table for quantity and size columns.")
            else:
                st.subheader("Consolidated Output")
                st.dataframe(df_out)
                summary['total_after_consolidation'] = int(df_out['qty'].sum())
                summary['widths_60_inch_created'] = int(df_out[df_out['width_final'] == 60]['qty'].sum())
                leftovers = df_out[df_out['width_final'] != 60].groupby('width_final')['qty'].sum().to_dict()
                summary['leftover_rolls'] = {int(k): int(v) for k, v in leftovers.items()}

                st.json(summary)

                csv = df_out.to_csv(index=False).encode('utf-8')
                st.download_button("Download consolidated CSV", csv, file_name="consolidated_rolls.csv", mime='text/csv')

    else:
        st.info("No table detected yet. Upload an Excel/CSV or paste the extracted table JSON/CSV/text above, then re-run parsing.")
        if pasted:
            st.caption("If your pasted data didn't parse automatically, it might not be in a strictly tabular format. Consider exporting to CSV/Excel from your OCR/table extractor first.")

    st.markdown("---")
    st.markdown("Notes: This page uses rule-based parsing and a deterministic consolidation algorithm. If your vendor columns use non-standard names or the OCR extraction is noisy, cleaning upstream or using an LLM-assisted parse may improve results. You can integrate your Gemini/LLM parsing step to produce a clean CSV/JSON that this page consumes.")
