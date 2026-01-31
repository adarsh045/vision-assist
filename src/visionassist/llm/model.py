import os
import base64
import requests
from io import BytesIO
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image

from visionassist.logger import logger
from visionassist.config import ENVITRONMENT, ALLOWED_LABELS

os.environ["GEMINI_API_KEY"] = ENVITRONMENT["GEMINI_API_KEY"]

OLLAMA_URL = ENVITRONMENT["OLLAMA_API_URL"]
OLLAMA_API_KEY = ENVITRONMENT["OLLAMA_API_KEY"]

class Label(BaseModel):
    name: str | None = Field(description="A label from the set of allowed labels.")


try:
    client = genai.Client()
except Exception as e:
    logger.info(f"Error initializing client: {e}. Ensure the API key is correct.")
    exit()

class LLMBaseModel:

    def __init__(self, model: str):
        self.model = model
    
    def predict_label(text:str):
        pass

    def analyze_image(image_path: str, object_name: str):
        pass

class Gemini(LLMBaseModel):

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        super().__init__(model)

    def predict_label(self, text:str):
        system_instruction = (
            "You are an exceptionally precise assistant. "
            "Return ONLY valid JSON that conforms to the given schema. "
            "Do not return plain text."
            "You can only choose a label from the following allowed labels: "
            "In case the label is not in the allowed labels, return null for the name field."
        )

        user_prompt = (
            f"User Text: {text}\n"
            f"Allowed Labels: {ALLOWED_LABELS}"
        )

        try:
            response = client.models.generate_content(
            model=self.model,
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_json_schema=Label.model_json_schema()
                )
            )

            label = Label.model_validate_json(response.text)
            
            return label.name
        except Exception as e:
            print(f"API Request Failed: {e}")
            return None

    def analyze_image(self, image_path: str, object_name: str):

        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return

        system_instruction = (
            "You are a helpful visual assistant.\n"
            "Your task is to help a user locate a requested object in an image as if you were physically pointing it out to them.\n"
            "Rules:\n"
            "- Respond in 2-3 short sentences only.\n"
            "- Do NOT use headings, lists, or formatting.\n"
            "- Do NOT explain your reasoning.\n"
            "- Use simple, natural language.\n"
            "- Mention only the most obvious identifying detail if needed (e.g., color or title).\n"
            "- Focus on relative location using nearby landmarks.\n"
            "- Sound like a human giving directions, not a report."
        )

        user_prompt = (
            f"Based on this image, please help the user with his query "
            f"I am not able to find my {object_name} could you suggest where it might be located in my home?"
        )

        contents = [img, user_prompt]

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )

            return response.text
        except Exception as e:
            print(f"API Request Failed: {e}")

class Ollama(LLMBaseModel):
    def __init__(self, model: str = "mistral:7b", image_model: str ="qwen3-vl:8b"):
        super().__init__(model)
        self.image_model = image_model

    def predict_label(self, text:str):
        system_prompt = (
            "You are an exceptionally precise assistant.\n"
            "You MUST return ONLY valid JSON.\n"
            "The JSON MUST match this schema exactly:\n"
            '{"name": "<label>"}\n'
            f"You can only choose from these allowed labels:\n{ALLOWED_LABELS}\n"
            "In case the label is not in the allowed labels, return null for the name field.\n"
            "Do NOT add explanations, markdown, or extra text."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Text: {text}"}
            ],
            "temperature": 0,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }

        try:
            r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json=payload, headers=headers, timeout=60)
            r.raise_for_status()

            content = r.json()["choices"][0]["message"]["content"]

            label = Label.model_validate_json(content)
            return label.name

        except Exception as e:
            logger.error(f"Ollama label prediction failed: {e}")
            return None

    def analyze_image(self, image_path: str, object_name: str):
        try:
            img = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None

        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        system_instruction = (
            "You are a helpful visual assistant.\n"
            "Your task is to help a user locate a requested object in an image as if you were physically pointing it out to them.\n"
            "Rules:\n"
            "- Respond in 2-3 short sentences only.\n"
            "- Do NOT use headings, lists, or formatting.\n"
            "- Do NOT explain your reasoning.\n"
            "- Use simple, natural language.\n"
            "- Mention only the most obvious identifying detail if needed (e.g., color or title).\n"
            "- Focus on relative location using nearby landmarks.\n"
            "- Sound like a human giving directions, not a report."
        )

        user_prompt = (
            f"Based on this image, please help the user with his query. "
            f"I am not able to find my {object_name}. "
            f"Could you suggest where it might be located in my home?"
        )

        payload = {
            "model": self.image_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        }

        try:
            r = requests.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=120
            )
            r.raise_for_status()

            return r.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama image analysis failed: {e}")
            return None
        
def getModel(model_name: str) -> LLMBaseModel:
    if model_name == "gemini":
        return Gemini()
    elif model_name == "ollama":
        return Ollama()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")