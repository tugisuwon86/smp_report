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
        api_key=os.environ["HF_TOKEN"],
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
You are an AI assistant that extracts structured information from purchase order text.
Please return JSON with the following fields:

customer: {{
  "company_name": "",
  "address": "",
  "email": "",
  "phone": ""
}},
items: [
  {{
    "product": "",
    "description": "",
    "quantity": "",
    "rate": "",
    "amount": ""
  }}
]

Text to analyze:
{text}
Return ONLY valid JSON, nothing else.
"""
