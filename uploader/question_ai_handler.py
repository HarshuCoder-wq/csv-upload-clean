import os
import requests
import time
import random
from dotenv import load_dotenv

load_dotenv() 

MAX_RETRIES = 5
API_CALL_DELAY = float(os.getenv("API_CALL_DELAY", 5.0))  # Delay in seconds between calls


def ask_together_ai(question, for_user_id):
    time.sleep(API_CALL_DELAY)
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing TOGETHER_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post("https://api.together.ai/v1/chat/completions", json=data, headers=headers, timeout=15)
    response.raise_for_status()
    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()
    return f"[TogetherAI for user {for_user_id}] {question} → {answer}"


def ask_comet_ai(question, for_user_id):
    time.sleep(API_CALL_DELAY)
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        raise ValueError("Missing COMET_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "comet-llama3",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post("https://api.comet.ai/v1/chat/completions", json=data, headers=headers, timeout=15)
    response.raise_for_status()
    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()
    return f"[CometAI for user {for_user_id}] {question} → {answer}"


def ask_aiml_ai(question, for_user_id):
    time.sleep(API_CALL_DELAY)
    api_key = os.getenv("AIML_API_KEY")
    if not api_key:
        raise ValueError("Missing AIML_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    response = requests.post("https://api.aimlapi.com/v1/chat/completions", json=data, headers=headers, timeout=15)
    response.raise_for_status()
    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()
    return f"[AIMLAI for user {for_user_id}] {question} → {answer}"


def try_with_retries(func, question, user_id, provider_name):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(question, user_id)
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            if status_code == 429:
                print(f"[RATE LIMIT] {provider_name} attempt {attempt} got 429 for user {user_id}. Backing off longer.")
                retry_after = int(http_err.response.headers.get("Retry-After", 10))  # fallback 10s
                time.sleep(retry_after + random.uniform(1, 3))
            else:
                print(f"[ERROR] {provider_name} attempt {attempt} failed for user {user_id}: {http_err}")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        except Exception as e:
            print(f"[ERROR] {provider_name} attempt {attempt} failed for user {user_id}: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 1))
    return None

def get_answer_from_ai(question, for_user_id):
    for func, name in [
        (ask_together_ai, "TogetherAI"),
        (ask_comet_ai, "CometAI"),
        (ask_aiml_ai, "AIMLAI")
    ]:
        result = try_with_retries(func, question, for_user_id, name)
        if result:
            return result
    return f"[ERROR] All AI services failed after {MAX_RETRIES} retries each for user {for_user_id}"
