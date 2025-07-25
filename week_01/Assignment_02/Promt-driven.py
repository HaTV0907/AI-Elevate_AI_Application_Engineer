import os
from openai import AzureOpenAI
 
# Step 1: Mock Input Data
task_descriptions = [
    "Install the battery module in the rear compartment, connect to the high-voltage harness, and verify torque on fasteners.",
    "Calibrate the ADAS (Advanced Driver Assistance Systems) radar sensors on the front bumper using factory alignment targets.",
    "Apply anti-corrosion sealant to all exposed welds on the door panels before painting.",
    "Perform leak test on coolant system after radiator installation. Record pressure readings and verify against specifications.",
    "Program the infotainment ECU with the latest software package and validate connectivity with dashboard display."
]
 
# Step 2: OpenAI Azure Client Setup
# Ensure these environment variables are set before running the script:
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://aiportalapi.stu-platform.live/jpe"
os.environ["AZURE_OPENAI_API_KEY"] = ""
client = AzureOpenAI(
    api_version="2024-07-01-preview", # Use the latest stable API version or the one specified
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
deployment_name = "GPT-4o-mini" # Or your specific deployment name, e.g., "gpt-35-turbo", "gpt-4"
 
def generate_instruction(task: str) -> str:
    """
    Generates step-by-step work instructions for a given manufacturing task
    using Azure ChatOpenAI.
    """
    # Key to the solution: The carefully crafted prompt.
    # We define the LLM's "persona" and the desired output format.
    prompt = f"""
    You are an expert automotive manufacturing supervisor, safety officer, and quality inspector combined.
    Your goal is to generate extremely clear, concise, and safe step-by-step work instructions
    for an assembly line worker, technician, or quality inspector.
 
    For each instruction, include the following:
    1.  **Safety Precautions:** Explicitly state any necessary PPE or safety procedures *before* the relevant step.
    2.  **Required Tools/Equipment:** Mention specific tools or equipment needed for the step.
    3.  **Detailed Actions:** Break down the task into logical, easy-to-follow, numbered steps.
    4.  **Acceptance Criteria/Verification:** How to confirm the step was successfully completed, including required readings or checks.
 
    Format the output as a numbered list. Be highly specific and assume the worker needs precise guidance.
 
    Task: \"\"\"{task}\"\"\"
 
    Work Instructions:
    """
   
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a highly detailed and safety-conscious automotive manufacturing expert."}, # System message to set the tone
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic and focused output
            max_tokens=500 # Limit output length to prevent overly verbose instructions
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while generating instructions for task '{task}': {e}")
        return "Failed to generate instructions."
 
# Step 3: Example Run
if __name__ == "__main__":
    print("Generating Work Instructions for New Car Model Tasks...\n")
    for i, task in enumerate(task_descriptions):
        print(f"--- Processing Task {i+1}/{len(task_descriptions)} ---")
        print(f"Task Description: {task}")
        instructions = generate_instruction(task)
        print(f"Generated Work Instructions:\n{instructions}\n")
        print("-" * 50) # Separator