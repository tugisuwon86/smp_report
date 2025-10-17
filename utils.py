# pdf_utils.py
import pdfplumber
import streamlit as st

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# llm_utils.py
import os
import requests
import openai
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = st.secrets['hugging-face']['api_token']

from openai import OpenAI

def query_openai(prompt):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

    # completion = client.chat.completions.create(
    #     model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    #     messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         ],
    # )
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b:groq",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message

def build_prompt(text):
    return f"""
You are an AI assistant that extracts structured data from purchase order (PO) documents.

Your task is to accurately parse the text and return structured JSON output containing vendor and shipping information, along with detailed item lines.

---

### 📦 Extraction Rules

1. **Section Identification**
   - The PO text may contain multiple columns (e.g., "VENDOR" on the left and "SHIP TO" on the right).
   - Treat each section independently based on its header:
     - The section labeled **"VENDOR"**, **"VENDOR/BILL TO"**, or **"BILL TO"** should be parsed under `"bill_to_customer"`.
     - The section labeled **"SHIP TO"** or **"SHIPPING"** should be parsed under `"ship_to_customer"`.
   - Each section typically includes:
     - Name or company
     - Address lines
     - City, State, ZIP
     - Optional email or phone number

   ⚠️ Do NOT merge information across columns or sections — extract content directly under the respective header.

2. **Item Table Extraction**
   - Items appear in a table format (rows and columns).
   - Each row represents one product/item.
   - Never merge or concatenate information from multiple rows or across columns.
   - Map fields as follows:
     - **product** → value under "Item" or "Product" column.
     - **description** → value under "Description" or "SKU" column (contains product details or identifiers).
     - **quantity** → integer quantity value (look for whole numbers).
     - **amount** → total price, often decimal or currency (may be missing).
   - If a column is missing, leave the field as an empty string.

3. **No Column Labels Case**
   - If the table has no headers, infer the meaning by position and content pattern:
     - Quantities are integers.
     - Rate and Amount are usually decimals or currency values.

4. **Output Requirements**
   - Return **only valid JSON**, with the structure below.
   - Do not include explanations, text, or markdown formatting.

---

### ✅ JSON Output Format

{{
  "bill_to_customer": {{
    "company_name": "",
    "address": "",
    "email": "",
    "phone": ""
  }},
  "ship_to_customer": {{
    "company_name": "",
    "address": "",
    "email": "",
    "phone": ""
  }},
  "items": [
    {{
      "product": "",
      "description": "",
      "quantity": "",
      "amount": ""
    }}
  ]
}}

---

### 🔍 Text to Analyze
{text}

Return ONLY valid JSON output.
"""
