import os
import csv
import json
import random
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIError

# Step 1: Load Azure credentials from .env file
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview"
)
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# Step 2: Define delay categories
DELAY_CATEGORIES = [
    "Traffic",
    "Customer Issue",
    "Vehicle Issue",
    "Weather",
    "Sorting/Labeling Error",
    "Human Error",
    "Technical System Failure",
    "Other"
]

# Step 3: Generate dummy log data into CSV
def generate_dummy_logs(filename="logs.csv"):
    samples = [
        "Heavy traffic near depot",
        "Customer wasn't available at drop-off",
        "Vehicle broke down mid-delivery",
        "Thunderstorm delayed departure",
        "Barcode unreadable, needed manual entry",
        "Driver missed turn and rerouted",
        "Arrived on schedule, no delay",
        "Wrong address on package",
        "System reboot caused docking issue",
        "Accident near warehouse caused delay",
        "Late start due to staff confusion",
        "Windstorm disrupted outdoor loading",
        "Scanner error during inventory",
        "Package mixed with wrong batch",
        "Driver forgot to confirm arrival",
        "Navigation system crashed",
        "Customer changed delivery time last minute",
        "Engine warning light triggered mid-trip",
        "Rain made roadside unsafe",
        "Security check took longer than expected"
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["log_id", "log_entry"])
        for i, entry in enumerate(samples, start=1):
            writer.writerow([i, entry])

# Step 4: Heuristic Classifier
def initial_classify(text):
    keywords = {
        "traffic": "Traffic",
        "accident": "Traffic",
        "customer": "Customer Issue",
        "unavailable": "Customer Issue",
        "engine": "Vehicle Issue",
        "vehicle": "Vehicle Issue",
        "rain": "Weather",
        "wind": "Weather",
        "storm": "Weather",
        "label": "Sorting/Labeling Error",
        "barcode": "Sorting/Labeling Error",
        "manual": "Sorting/Labeling Error",
        "wrong": "Human Error",
        "missed": "Human Error",
        "forgot": "Human Error",
        "system": "Technical System Failure",
        "glitch": "Technical System Failure",
        "crashed": "Technical System Failure",
        "scanner": "Technical System Failure"
    }
    for k, v in keywords.items():
        if k in text.lower():
            return v
    return "Other"

# Step 5: LLM Refinement using Azure
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_random_exponential(min=1, max=5),
    stop=stop_after_attempt(5),
    reraise=True
)
def refine_classification(text, initial_label):
    prompt = f"""
You are a logistics assistant. A log entry has been auto-categorized as "{initial_label}".
Please confirm or correct it by choosing one of the following categories:
- Traffic
- Customer Issue
- Vehicle Issue
- Weather
- Sorting/Labeling Error
- Human Error
- Technical System Failure
- Other

Log Entry:
\"\"\"{text}\"\"\"

Return only the most appropriate category from the list above.
"""
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Step 6: Classification Pipeline
def classify_logs(filename="logs.csv"):
    results = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            log_id = row["log_id"]
            log_entry = row["log_entry"]
            initial = initial_classify(log_entry)
            final = refine_classification(log_entry, initial)
            results.append({
                "log_id": log_id,
                "log_entry": log_entry,
                "initial_label": initial,
                "final_label": final
            })
    return results

# Step 7: Run end-to-end
if __name__ == "__main__":
    generate_dummy_logs()
    print("🚀 Dummy data generated and saved to logs.csv")
    print("🔎 Starting classification...")
    final_results = classify_logs()
    print("\n📊 Classification Results:\n")
    for result in final_results:
        print(f"ID {result['log_id']}:")
        print(f"  Log: {result['log_entry']}")
        print(f"  Initial: {result['initial_label']}")
        print(f"  Final:   {result['final_label']}")
        print("-" * 50)