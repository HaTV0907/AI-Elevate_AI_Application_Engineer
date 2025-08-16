
import os
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

class LlamaModel:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH")
        context_size = int(os.getenv("CONTEXT_SIZE", 512))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        try:
            self.llm = Llama.from_pretrained(
                                            repo_id="HeRksTAn/Meta-Llama-3-8B-Instruct-Q4_K_S-GGUF",
                                            filename="meta-llama-3-8b-instruct-q4_k_s.gguf",)

            print(f"✅ LLaMA model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_text(self, prompt: str) -> str:
        max_tokens = int(os.getenv("MAX_TOKENS", 300))
        temperature = float(os.getenv("TEMPERATURE", 0.7))
        try:
            output = self.llm.create_chat_completion(prompt, max_tokens=max_tokens, temperature=temperature)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

class ResumeGenerator:
    def __init__(self, model: LlamaModel):
        self.model = model

    def build_prompt(self, user_data: dict) -> str:
        prompt = (
            "Generate a professional resume:\n"
            f"Name: {user_data['name']}\n"
            f"Email: {user_data['email']}\n"
            f"Phone: {user_data['phone']}\n"
            f"Summary: {user_data['summary']}\n"
            f"Skills: {', '.join(user_data['skills'])}\n"
            f"Experience:\n"
        )
        for exp in user_data['experience']:
            prompt += f"- {exp['role']} at {exp['company']} ({exp['duration']}): {exp['description']}\n"
        prompt += f"Education: {user_data['education']}\n"
        prompt += "Format the resume professionally."
        return prompt

    def generate_resume(self, user_data: dict) -> str:
        prompt = self.build_prompt(user_data)
        return self.model.generate_text(prompt)
