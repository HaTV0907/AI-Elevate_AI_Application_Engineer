import streamlit as st
import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Azure OpenAI config
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Embedding config
AZURE_OPENAI_API_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_API_EMBEDDING_KEY")
AZURE_OPENAI_API_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_API_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Streamlit UI
st.title("🧪 Blood Test Analyzer")
uploaded_file = st.file_uploader("Upload a blood test image (PNG or JPEG)", type=["png", "jpeg", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if image.format not in ["PNG", "JPEG"]:
        st.error("Only PNG or JPEG images are supported.")
        st.stop()

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Prepare messages for vision model
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
        # Step 1: Vision model request
        client = openai.OpenAI(
            base_url=AZURE_OPENAI_API_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
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

        # Step 2: Embedding using OpenAI SDK
        embed_client = openai.OpenAI(
            base_url=AZURE_OPENAI_API_EMBEDDING_ENDPOINT,
            api_key=AZURE_OPENAI_API_EMBEDDING_KEY
        )

        embed_response = embed_client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            input=[explanation]  # ✅ Correct format: list of strings
        )

        embedding = embed_response.data[0].embedding

        # Step 3: Store in Pinecone
        index.upsert([
            {
                "id": uploaded_file.name,
                "values": embedding,
                "metadata": json_data
            }
        ])

        st.success("✅ Data stored in Pinecone!")

    except Exception as e:
        st.error("🚨 An error occurred during processing.")
        st.exception(e)
