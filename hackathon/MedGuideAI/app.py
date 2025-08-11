import streamlit as st
import io
import base64
from PIL import Image
import fitz  # PyMuPDF
import openai
import json
import os
from utils import load_config
from vectorDB import init_pinecone, embed_text, store_in_pinecone
import difflib

# Load config
config = load_config()

# Init Pinecone
index = init_pinecone(
    api_key=config["PINECONE_API_KEY"],
    env=config["PINECONE_ENV"],
    index_name=config["PINECONE_INDEX_NAME"]
)

def highlight_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        fromfile='Current Report',
        tofile='Matched Report',
        lineterm=''
    )
    return '\n'.join(diff)

def generate_health_advice(diff_text):
    prompt = f"""You're a medical assistant. Based on the following differences between two blood test reports:\n{diff_text}\n
                 Give personalized advice about diet and exercise to improve health outcomes."""
    
    client = openai.OpenAI(
        base_url=config["AZURE_OPENAI_API_ENDPOINT"],
        api_key=config["AZURE_OPENAI_API_KEY"]
    )

    response = client.chat.completions.create(
        model=config["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )

    return response.choices[0].message.content

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

def save_id_locally(file_id):
    path = "stored_ids.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            ids = json.load(f)
    else:
        ids = []

    if file_id not in ids:
        ids.append(file_id)
        with open(path, "w") as f:
            json.dump(ids, f)


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
    with open("stored_ids.json", "r") as f:
        report_ids = json.load(f)
except:
    report_ids = []

if report_ids:
    selected_id = st.sidebar.selectbox("Select a report to view", report_ids)
    if selected_id:
        selected_meta = index.fetch([selected_id]).vectors[selected_id].metadata
        st.subheader(f"📄 Viewing: {selected_id}")
        st.markdown(f"**Filename:** `{selected_meta.get('filename')}`")
        st.markdown(f"**Source:** `{selected_meta.get('source')}`")
        st.text_area("Explanation", selected_meta.get("explanation", ""), height=300)
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
            save_id_locally(uploaded_file.name)
            st.success("✅ Data stored in Pinecone!")

            # Save latest report in session
            st.session_state["latest_report"] = {
                "filename": uploaded_file.name,
                "explanation": explanation,
                "source": source_type
            }

            # --- Search and Compare ---
            st.subheader("🔍 Search Past Reports to Compare")

            search_query = st.text_input("Search by filename or keyword")

            # Load stored report IDs
            try:
                with open("stored_ids.json", "r") as f:
                    all_ids = json.load(f)
            except:
                all_ids = []

            # Filter reports by search query
            matching_ids = []
            for rid in all_ids:
                meta = index.fetch([rid]).vectors[rid].metadata
                if search_query.lower() in rid.lower() or search_query.lower() in meta.get("explanation", "").lower():
                    matching_ids.append(rid)

            if matching_ids:
                selected_compare_id = st.selectbox("Select a report to compare", matching_ids)
                if selected_compare_id:
                    compare_meta = index.fetch([selected_compare_id]).vectors[selected_compare_id].metadata
                    st.subheader("🆚 Comparison")
                    diff_text = highlight_diff(explanation, compare_meta.get("explanation", ""))
                    st.subheader("🔍 Differences")
                    st.code(diff_text, language='diff')
                    advice = generate_health_advice(diff_text)
                    st.subheader("💡 Diet & Exercise Advice")
                    st.markdown(advice)
                    st.markdown(f"**Current Source:** `{source_type}`")
                    st.markdown(f"**Matched Source:** `{compare_meta.get('source', '')}`")
            else:
                if search_query:
                    st.warning("No matching reports found.")
    except Exception as e:
        st.error("🚨 An error occurred during processing.")
        st.exception(e)
