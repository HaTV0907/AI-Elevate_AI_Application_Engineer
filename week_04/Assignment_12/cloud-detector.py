import os
import base64
import requests
from PIL import Image
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# --- Setup LLM ---
# Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# --- Output Schema ---
class WeatherResponse(BaseModel):
    accuracy: float = Field(description="The accuracy of the result")
    result: str = Field(description="The result of the classification")

llm_with_structured_output = llm.with_structured_output(WeatherResponse)

# --- Dummy Image URLs ---
image_urls = [
    "https://images.pexels.com/photos/53594/blue-clouds-day-fluffy-53594.jpeg",
    "https://images.pexels.com/photos/158163/clouds-cloudporn-weather-lookup-158163.jpeg",
    "https://images.pexels.com/photos/110874/pexels-photo-110874.jpeg"
]

# --- Inference Loop ---
for idx, image_url in enumerate(image_urls):
    print(f"\n🔍 Processing Image {idx+1}: {image_url}")
    try:
        response = requests.get(image_url)
        image_bytes = response.content
        image_data_base64 = base64.b64encode(image_bytes).decode("utf-8")

        message = [
            {
                "role": "system",
                "content": """Based on the satellite image provided, classify the scene as either:
                'Clear' (no clouds) or 'Cloudy' (with clouds).
                Respond with only one word: either 'Clear' or 'Cloudy' and Accuracy.
                Do not provide explanations.""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify the scene as either: 'Clear' or 'Cloudy' and Accuracy."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"}},
                ],
            },
        ]

        result = llm_with_structured_output.invoke(message)
        print(f"✅ Prediction: {result.result}")
        print(f"📊 Accuracy: {result.accuracy:.2f}%")

    except Exception as e:
        print(f"❌ Error processing image: {e}")
