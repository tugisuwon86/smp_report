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
import json
import base64
import time
import pandas as pd
from io import BytesIO

load_dotenv()
HF_TOKEN = st.secrets['hugging-face']['api_token']
GEMINI_TOKEN = st.secrets['gemini-api']['api_token']

### image files
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key="

# --- Extraction Prompts (Same as your original script) ---

SYSTEM_PROMPT = (
    "You are an expert data extraction and analysis bot. Your task is to analyze the provided image "
    "(which contains structured data, likely a table or list). You must extract all text and structure "
    "the data into a clean, comprehensive Markdown table format, using appropriate headings and columns based on "
    "the content. Do not include any introductory or concluding sentences, only the Markdown table."
)

USER_QUERY = (
    "Analyze the content of this image. Recreate the table structure and its data completely and "
    "accurately in a Markdown table format. Ensure every piece of information is captured."
)

# --- Core Utility Functions ---

def read_file_to_base64(uploaded_file):
    """Reads Streamlit's UploadedFile object and returns its Base64 encoded string and MIME type."""
    # Use uploaded_file.getvalue() to get the bytes
    encoded_string = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    mime_type = uploaded_file.type
    return encoded_string, mime_type

def call_gemini_api(base64_data: str, mime_type: str) -> str:
    """Calls the Gemini API with exponential backoff for image analysis."""
    headers = {'Content-Type': 'application/json'}
    api_url = API_URL_TEMPLATE + GEMINI_TOKEN

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": USER_QUERY},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_data
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        }
    }

    max_attempts = 1
    for attempt in range(max_attempts):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # Successful response
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            extracted_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No text extracted.')
            
            return extracted_text

        except requests.exceptions.RequestException as e:
            if attempt < max_attempts - 1 and (response.status_code == 429 or response.status_code >= 500):
                delay = (2 ** attempt) + (random.random() * 1) # Exponential backoff + jitter
                time.sleep(delay)
            else:
                st.error(f"Critical API Error: {e}")
                return "Extraction failed due to a critical API error."
        
    return "Extraction failed after multiple retries."

def convert_markdown_to_dataframe(markdown_table: str) -> pd.DataFrame:
    """
    Converts a Markdown table string (separated by '|') to a Pandas DataFrame.
    Handles cleanup for leading/trailing pipes and separator lines.
    """
    # Split the string into lines
    lines = markdown_table.strip().split('\n')

    # Check for minimal table structure
    if len(lines) < 2 or '---' not in markdown_table:
        # If the LLM returns non-tabular data, wrap it in a single-column DataFrame
        if not markdown_table.strip():
            return pd.DataFrame()
        return pd.DataFrame({"Result": [markdown_table.strip()]})

    # 1. Filter out the separator line (which contains '---')
    lines_without_separator = [line for line in lines if '---' not in line]

    # 2. Join the remaining lines and wrap them in StringIO
    clean_markdown = "\n".join(lines_without_separator)
    
    # 3. Read the cleaned data using '|' as a separator
    # skipinitialspace=True cleans up spaces around the pipes
    df = pd.read_csv(StringIO(clean_markdown), sep='|', skipinitialspace=True)
    
    # 4. Clean up the DataFrame
    # Drop columns that are entirely empty (these result from leading/trailing pipes)
    df = df.dropna(axis=1, how='all')
    
    # Reset column names (strip whitespace from column headers)
    df.columns = df.columns.str.strip()
    
    return df
    
### PDF files
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
        # model="openai/gpt-oss-120b:groq",
        model="openai/gpt-oss-20b:groq",
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
   - Please ignore the top section. Look for the header related to **"VENDOR"** or **"BILL TO"** or **"SHIP TO"** first.
   - Treat each section independently based on its header:
     - The section labeled **"VENDOR"**, **"VENDOR/BILL TO"**, or **"BILL TO"** should be parsed under `"bill_to_customer"`.
     - The section labeled **"SHIP TO"** or **"SHIPPING"** should be parsed under `"ship_to_customer"`.
     - If you find "SMP Corporation" as company under shipping, that's wrong. You need to swap vendor and ship_to information including address, email, and phone.
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
     - **description** → value under "Description" or "SKU" column (contains product details or identifiers). It should not contain space. It is unique identify that may have hyphen. If the string is broken by colon, only print the last element with colon delimiter.
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
