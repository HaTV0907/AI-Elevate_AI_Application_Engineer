import streamlit as st
import io
import base64
from PIL import Image
import fitz  # PyMuPDF
import openai

from utils import load_config
from vectorDB import init_pinecone, embed_text, store_in_pinecone

# Load config
config = load_config()

# Init Pinecone
index = init_pinecone(
    api_key=config["PINECONE_API_KEY"],
    env=config["PINECONE_ENV"],
    index_name=config["PINECONE_INDEX_NAME"]
)

# --- Helper Functions ---
def process_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def process_image(file):
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if image.format not in ["PNG", "JPEG"]:
        st.error("Only PNG or JPEG images are supported.")
        return None

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Extract key blood test fields from the image and explain them."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}"
                    }
                }
            ]
        }
    ]

    client = openai.OpenAI(
        base_url=config["AZURE_OPENAI_API_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"]
    )

    response = client.chat.completions.create(
        model=config["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content

def explain_text(text):
    client = openai.OpenAI(
        base_url=config["AZURE_OPENAI_API_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"]
    )

    response = client.chat.completions.create(
        model=config["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant. Explain the blood test results or medical information found in this PDF."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content

# --- UI ---
st.set_page_config(page_title="🧪 Blood Test Analyzer", layout="wide")
st.title("🧪 Blood Test Analyzer")

# Sidebar: Browse stored reports
st.sidebar.title("📁 Past Reports")
try:
    index_stats = index.describe_index_stats()
    namespaces = index_stats.namespaces or {}
    default_namespace = namespaces.get("") or {}
    report_ids = list(default_namespace.keys()) if isinstance(default_namespace, dict) else []
except Exception:
    report_ids = []

if report_ids:
    selected_id = st.sidebar.selectbox("Select a report to view", report_ids)
    if selected_id:
        selected_meta = index.fetch([selected_id]).vectors[selected_id].metadata
        st.sidebar.markdown(f"**Filename:** {selected_meta.get('filename')}")
        st.sidebar.markdown(f"**Source:** {selected_meta.get('source')}")
        st.sidebar.text_area("Explanation", selected_meta.get("explanation", ""), height=200)
else:
    st.sidebar.write("🗂️ No reports found yet.")

# Upload section
uploaded_file = st.file_uploader("Upload a blood test image or PDF report", type=["png", "jpeg", "jpg", "pdf"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            st.subheader("📄 PDF Preview")
            pdf_text = process_pdf(uploaded_file)
            st.text_area("Extracted Text", pdf_text, height=300)

            if not pdf_text.strip():
                st.error("❌ No readable text found in PDF.")
                st.stop()

            explanation = explain_text(pdf_text)
            source_type = "pdf"

        else:
            explanation = process_image(uploaded_file)
            source_type = "image"

        if explanation:
            st.subheader("🧾 Explanation")
            st.write(explanation)

            json_data = {
                "filename": uploaded_file.name,
                "explanation": explanation,
                "source": source_type
            }

            st.subheader("📦 JSON Output")
            st.json(json_data)

            embedding = embed_text(
                text=explanation,
                endpoint=config["AZURE_OPENAI_API_EMBEDDING_ENDPOINT"],
                api_key=config["AZURE_OPENAI_API_EMBEDDING_KEY"],
                model=config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
            )

            store_in_pinecone(index, uploaded_file.name, embedding, json_data)
            st.success("✅ Data stored in Pinecone!")

    except Exception as e:
        st.error("🚨 An error occurred during processing.")
        st.exception(e)
