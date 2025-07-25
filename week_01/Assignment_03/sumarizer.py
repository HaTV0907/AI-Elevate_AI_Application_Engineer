import os
import argparse
from openai import AzureOpenAI
import textwrap # For handling long texts
from dotenv import load_dotenv # if using .env file
# need to export these env variables
# --- Configuration (from Environment Variables) ---
API_VERSION = "2024-07-01-preview" # Ensure this matches your Azure OpenAI API version
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # Make sure this environment variable is set
 
load_dotenv() # if using .env file
 
# ... (rest of your imports)
 
# --- Configuration (from Environment Variables) ---
API_VERSION = "2024-07-01-preview"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
 
print(f"DEBUG: AZURE_OPENAI_ENDPOINT = '{AZURE_OPENAI_ENDPOINT}'")
print(f"DEBUG: AZURE_OPENAI_API_KEY = '{AZURE_OPENAI_API_KEY[:5]}...{AZURE_OPENAI_API_KEY[-5:]}'" if AZURE_OPENAI_API_KEY else "DEBUG: AZURE_OPENAI_API_KEY is not set.")
print(f"DEBUG: DEPLOYMENT_NAME = '{DEPLOYMENT_NAME}'")
 
# Check if essential environment variables are set
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not DEPLOYMENT_NAME:
    print("Error: Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME environment variables.")
    exit(1)
 
# --- Step 1: Azure OpenAI Client Setup ---
try:
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    print("Please check your AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    exit(1)
 
# --- Function to call the LLM for summarization ---
def summarize_text_with_gpt(text_chunk: str) -> str:
    """
    Calls the Azure OpenAI ChatCompletion API to summarize a given text chunk.
    """
    prompt = f"""You are a helpful assistant specialized in summarizing meeting notes.
    Summarize the following meeting transcript chunk with key points, decisions, and action items.
    Focus on extracting the most critical information concisely.
   
    Transcript Chunk:
    \"\"\"
    {text_chunk}
    \"\"\"
   
    Summary:
    """
   
    messages = [
        {"role": "system", "content": "You are an expert summarizer for business meetings, focusing on key takeaways, decisions, and action items."},
        {"role": "user", "content": prompt}
    ]
 
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME, # Use the deployment name here
            messages=messages,
            temperature=0.3,     # Lower temperature for more factual, less creative summaries
            max_tokens=700       # Adjust based on desired summary length and model capacity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Catch specific API errors for better debugging
        if "401" in str(e) or "authentication" in str(e).lower():
            print(f"Authentication/Authorization Error: Check your API key and deployment name. Detail: {e}")
        elif "rate limit" in str(e).lower():
            print(f"Rate Limit Exceeded: You're sending too many requests too quickly. Detail: {e}")
        elif "context_length_exceeded" in str(e).lower():
            print(f"Context Length Exceeded (Chunking Issue): Even after chunking, a piece might be too large. Detail: {e}")
        else:
            print(f"An unexpected API error occurred: {e}")
        return "Failed to generate summary for this chunk."
 
# --- Strategy for Handling Large Transcripts ---
def summarize_long_transcript(transcript: str, max_chunk_tokens: int = 4000) -> str:
    """
    Summarizes a long transcript by splitting it into chunks, summarizing each chunk,
    and then combining/summarizing the individual summaries.
   
    `max_chunk_tokens` should be significantly less than your model's total context window
    (e.g., gpt-4o-mini has 128k, so 4k leaves plenty of room for prompt and output).
    """
   
    # A simple token-based chunking might be needed here.
    # For simplicity, we'll use character-based chunking with an approximation for tokens.
    # A rough estimate: 1 token is about 4 characters.
    # So, for 4000 tokens, we'd aim for roughly 16000 characters.
    # This is a simplification; for production, use a proper token counter.
   
    approx_char_limit = max_chunk_tokens * 4
   
    # Use textwrap.wrap for basic splitting, trying to break on logical paragraphs/sentences
    # rather than just hard character limits.
    # For very large files, a more sophisticated splitter (e.g., from LangChain) would be better.
   
    chunks = []
    current_chunk = ""
    for paragraph in transcript.split('\n\n'): # Split by double newline to get paragraphs
        if len(current_chunk) + len(paragraph) + 2 < approx_char_limit: # +2 for potential newlines
            current_chunk += (paragraph + "\n\n")
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
 
    print(f"Transcript split into {len(chunks)} chunks.")
   
    individual_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary_chunk = summarize_text_with_gpt(chunk)
        if "Failed to generate summary" not in summary_chunk:
            individual_summaries.append(f"Summary of Part {i+1}:\n{summary_chunk}\n")
        else:
            print(f"Skipping chunk {i+1} due to error.")
 
    if not individual_summaries:
        return "Could not generate any summaries from the transcript."
 
    # If there's only one chunk or the combined summaries are short, just return that.
    combined_summaries_text = "\n".join(individual_summaries)
    if len(combined_summaries_text) < max_chunk_tokens * 3: # Arbitrary threshold for final summary
        print("Final summary generated from combined chunks.")
        return combined_summaries_text
   
    # If the combined summaries are still too long, summarize them again.
    print("\nIndividual summaries are still long. Performing a final meta-summary...")
    final_prompt = f"""Consolidate the following individual meeting segment summaries into one comprehensive summary.
    Ensure to include all key points, decisions, and action items from across all segments.
   
    Individual Summaries:
    \"\"\"
    {combined_summaries_text}
    \"\"\"
   
    Final Comprehensive Meeting Summary:
    """
   
    messages = [
        {"role": "system", "content": "You are an expert in synthesizing multiple summaries into one cohesive and comprehensive meeting overview."},
        {"role": "user", "content": final_prompt}
    ]
   
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during final meta-summarization: {e}")
        return "Failed to generate a comprehensive summary."
 
# --- Main Application Logic (CLI Interface) ---
def main():
    parser = argparse.ArgumentParser(description="Summarize a meeting transcript using Azure OpenAI GPT models.")
    parser.add_argument("transcript_file", type=str, help="Path to the meeting transcript text file.")
    args = parser.parse_args()
 
    transcript_file_path = args.transcript_file
 
    # Step 2: Load transcript text from the specified file
    try:
        with open(transcript_file_path, "r", encoding="utf-8") as file:
            transcript = file.read()
        if not transcript.strip():
            print("Error: The transcript file is empty.")
            return
        print(f"Successfully loaded transcript from: {transcript_file_path} (length: {len(transcript)} characters)")
    except FileNotFoundError:
        print(f"Error: Transcript file not found at '{transcript_file_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return
 
    # Step 3 & 4: Process and call OpenAI API for summarization
    print("Generating meeting summary...")
    summary = summarize_long_transcript(transcript)
 
    # Step 5: Extract and display summary
    print("\n" + "="*30)
    print("Meeting Summary:")
    print("="*30)
    print(summary)
    print("="*30 + "\n")
 
if __name__ == "__main__":
    main()