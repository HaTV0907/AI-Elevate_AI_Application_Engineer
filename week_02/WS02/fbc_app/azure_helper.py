import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
API_VERSION = "2024-07-01-preview" # Ensure this matches your Azure OpenAI API version
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# debug log
print("🔧 Using deployment name:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("🔧 API endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("🔧 API key loaded:", bool(os.getenv("AZURE_OPENAI_API_KEY")))

def explain_fbc_result(test_name, value, unit, normal_range):
    functions = [
        {
            "name": "explain_fbc_test",
            "description": "Explains an FBC test result",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "normal_range": {"type": "string"}
                },
                "required": ["test_name", "value", "unit", "normal_range"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a medical assistant who explains FBC test results in simple, patient-friendly language."},
        {"role": "function", "name": "explain_fbc_test", "content": json.dumps({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "normal_range": normal_range
        })}
    ]

    print(messages)
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    return response.choices[0].message.content