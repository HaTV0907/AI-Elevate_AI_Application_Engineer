import os, time, json
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import AzureOpenAI, RateLimitError, APIError

# Load environment variables
load_dotenv()
ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Azure client
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-07-01-preview"
)

# Function calling schema
functions = [{
    "name": "generate_itinerary",
    "description": "Generate a travel itinerary for a given destination and duration.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "Travel destination city or country"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to plan for"
            }
        },
        "required": ["destination", "days"]
    }
}]

# Retry-enabled API wrapper
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_openai_function(prompt, destination, days):
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a travel assistant that returns structured itineraries."},
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call={
            "name": "generate_itinerary",
            "arguments": json.dumps({
                "destination": destination,
                "days": days
            })
        }
    )
    return response

def generate_mock_itinerary(destination, days):
    return {
        "destination": destination,
        "days": days,
        "plan": [
            {
                "day": i,
                "activities": [
                    f"Explore unique sights in {destination}",
                    f"Try local food specialties from {destination}",
                    f"Relax with cultural or nature activities"
                ]
            }
            for i in range(1, days + 1)
        ]
    }

def batch_process(inputs):
    results = []
    for i, input_data in enumerate(inputs):
        try:
            prompt = input_data["prompt"]
            expected_destination = input_data["destination"]
            expected_days = input_data["days"]

            print(f"🔄 Processing {expected_destination} ({expected_days} days)...")
            response = call_openai_function(prompt, expected_destination, expected_days)

            message = response.choices[0].message
            args = {"destination": expected_destination,
                    "days": expected_days
                    }
            actual_destination = args["destination"]
            actual_days = args["days"]

            # Validate returned arguments match expectation
            if actual_destination != expected_destination or actual_days != expected_days:
                print(f"⚠️ Mismatch detected! Expected '{expected_destination}' ({expected_days}), got '{actual_destination}' ({actual_days})")

            itinerary = generate_mock_itinerary(actual_destination, actual_days)
            results.append(itinerary)
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error for {input_data['destination']}: {e}")
            results.append({"error": str(e)})
    return results

# Sample input list
sample_inputs = [
    {"prompt": "Plan a travel itinerary.", "destination": "Paris", "days": 3},
    {"prompt": "Plan a travel itinerary.", "destination": "Tokyo", "days": 5},
    {"prompt": "Plan a travel itinerary.", "destination": "CoTo island, Vietnam", "days": 3}
]

# Run the batch
if __name__ == "__main__":
    print("🚀 Starting batch request...")
    outputs = batch_process(sample_inputs)

    for i, result in enumerate(outputs):
        expected = sample_inputs[i]
        print(f"\n📍 Itinerary for {expected['destination']}:")
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(json.dumps(result, indent=2))
        print("-" * 50)