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


def query_mistral(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    url = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 400, "temperature": 0.2}
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list):
        return data[0]["generated_text"]
    return data

def build_prompt(text):
    return f"""
You are an AI assistant that extracts structured information from a purchase order (PO) text.
Your goal is to return all relevant information in **valid JSON format only**, no explanations or comments.

### Extraction Rules:

1. **Vendor / Bill To**
   - Look for sections labeled `Vendor`, `Bill To`, or similar headers. 
   - It is a column so the actual information is written below the section header
   - Extract this information under `bill_to_customer`.
   - Include company name, address, email, and phone number if available.

2. **Ship To**
   - Look for sections labeled `Ship To` or `Shipping Address`.
   - It is a column so the actual information is written below the section header
   - Extract this information under `ship_to_customer`.
   - Include company name, address, email, and phone number if available.

3. **Items / Products**
   - Each item corresponds to a line in the PO table.
   - Use cell breaker if possible meaning do not concatenate strings if it is from different cell/column.
   - Use the following mapping:
     - **product** → value under `Item`, `Product`, or similar field.
     - **description** → value under `Description` or `SKU` (which may contain a unique identifier or detailed info).
     - **quantity** → numeric value (integer) representing count of items.
     - **amount** → total price (decimal), if available.
   - If a field (rate/amount) is missing, leave it as an empty string.

4. **No Column Labels Case**
   - If the PO has no explicit headers, infer field meanings by analyzing context (e.g., quantity usually integer, rate/amount often contain decimals or currency symbols).

5. **Formatting**
   - Return strictly in valid JSON format (no markdown, comments, or text).
   - Structure must be:

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

### Text to Analyze:
{text}

Return ONLY valid JSON — do not include any extra commentary or explanations.
"""
