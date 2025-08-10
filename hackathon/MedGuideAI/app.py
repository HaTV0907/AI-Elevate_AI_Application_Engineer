import streamlit as st
import io
import base64
from PIL import Image
import openai
import fitz  # PyMuPDF

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

# Streamlit UI
st.title("🧪 Blood Test Analyzer")
uploaded_file = st.file_uploader(
    "Upload a blood test image or PDF report",
    type=["png", "jpeg", "jpg", "pdf"]
)

if uploaded_file and uploaded_file.type == "application/pdf":
    st.subheader("📄 PDF Preview")

    # Extract text from PDF
    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    st.text_area("Extracted Text", pdf_text, height=300)

    if not pdf_text.strip():
        st.error("❌ No readable text found in PDF.")
        st.stop()

    try:
        # Step 1: Ask GPT to explain PDF content
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
                    "content": pdf_text
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        explanation = response.choices[0].message.content
        st.subheader("🧾 Explanation")
        st.write(explanation)

        json_data = {
            "filename": uploaded_file.name,
            "explanation": explanation
        }

        st.subheader("📦 JSON Output")
        st.json(json_data)

        # Step 2: Embed and store in Pinecone
        embedding = embed_text(
            text=explanation,
            endpoint=config["AZURE_OPENAI_API_EMBEDDING_ENDPOINT"],
            api_key=config["AZURE_OPENAI_API_EMBEDDING_KEY"],
            model=config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
        )

        store_in_pinecone(index, uploaded_file.name, embedding, json_data)

        st.success("✅ PDF data stored in Pinecone!")

    except Exception as e:
        st.error("🚨 Error processing PDF.")
        st.exception(e)

elif uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if image.format not in ["PNG", "JPEG"]:
        st.error("Only PNG or JPEG images are supported.")
        st.stop()

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

    try:
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

        explanation = response.choices[0].message.content
        st.subheader("🧾 Explanation")
        st.write(explanation)

        json_data = {
            "filename": uploaded_file.name,
            "explanation": explanation
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
