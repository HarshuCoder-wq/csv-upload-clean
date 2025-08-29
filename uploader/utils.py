import mysql.connector
import os
from datetime import datetime
import re
import hashlib
import json
import requests
import time
import html

from urllib.parse import urlparse
from uuid import uuid4

# Utility functions that can be shared across modules

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

def fallback_generate_fields(title):
    """Fallback function to return basic metadata when AI fails."""
    return {
        "slug": slugify(title),
        "title_hash": hashlib.md5(title.encode()).hexdigest(),
        "keywords": f"{title.lower()}, fallback",
        "summary": f"This is a fallback summary for '{title}'.",
        "content": f"<p>This is fallback content generated when the AI did not return valid JSON for '{title}'.</p>"
    }

def log_ai_error(local_db_config, blog_title, provider_name, model_name, status_code, error_message, input_prompt, response, retry_attempt=0):
    """Logs AI generation errors to the blog_ai_error_logs table in the local database."""
    log_conn = None
    try:
        log_conn = mysql.connector.connect(**local_db_config)
        cursor = log_conn.cursor()
        cursor.execute("""
            INSERT INTO blog_ai_error_logs (blog_title, provider_name, model_name, status_code, error_message, input_prompt, response, retry_attempt, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (blog_title, provider_name, model_name, status_code, error_message, input_prompt, response, retry_attempt))
        log_conn.commit()
        print(f"📄 Logged AI error for '{blog_title}' to blog_ai_error_logs.")
    except Exception as e:
        print(f"❌ Failed to log AI error for '{blog_title}': {e}")
    finally:
        if log_conn:
            log_conn.close()

def _xml_escape(text):
    """Escapes special characters for XML."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\"", "&quot;")
    text = text.replace("'", "&apos;")
    return text

def chunk_list(data_list, chunk_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i + chunk_size]

# The safe_parse_json_response, call_ai_api_for_content, and call_ai_api_for_image
# functions were moved here from upload_posts.py/local_content_generator.py
# to centralize AI interaction logic.

def safe_parse_json_response(title, api_response_json, ai_provider_name, input_prompt_for_logging, local_db_config, model_name_for_logging):
    """
    Parses the raw JSON response object from an AI API.
    Handles specific structures for Together AI/Comet vs. AIML API.
    """
    import re
    import json

    generated_text = ""
    error_detail = ""

    try:
        normalized_provider = ai_provider_name.lower()

        if normalized_provider == 'aiml_api':
            if 'choices' in api_response_json and api_response_json['choices'] and \
               'message' in api_response_json['choices'][0] and \
               'content' in api_response_json['choices'][0]['message']:
                generated_text = api_response_json['choices'][0]['message']['content']
            else:
                error_detail = "AIML API response missing expected 'choices[0].message.content' structure."
                raise ValueError(error_detail)

        elif normalized_provider in ['together_ai', 'comet']:
            generated_text = api_response_json.get('choices', [{}])[0].get('text', '')
            if not generated_text:
                error_detail = "Together AI/Comet response missing 'text' field in choices."
                raise ValueError(error_detail)
        else:
            error_detail = f"Unsupported AI provider for content parsing: {ai_provider_name}"
            raise ValueError(error_detail)

        # --- Clean & Extract Raw JSON Text ---
        generated_text = generated_text.strip()
        generated_text = re.sub(r"^```json|```$", "", generated_text, flags=re.MULTILINE).strip()

        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        pure_json_string = generated_text[json_start:json_end].strip()

        # Sanitize problematic characters
        pure_json_string = pure_json_string.replace('\xa0', ' ')  # non-breaking spaces
        pure_json_string = pure_json_string.replace('\r', '')
        pure_json_string = pure_json_string.replace('\t', ' ')
        pure_json_string = pure_json_string.replace('\n', '')

        # Escape problematic quotes inside content field
        def escape_quotes(match):
            key = match.group(1)
            value = match.group(2).replace('"', '\\"')  # Escape inner quotes
            return f'"{key}": "{value}"'

        pure_json_string = re.sub(r'"(content)"\s*:\s*"([^"]*?)"', escape_quotes, pure_json_string)

        if not pure_json_string.startswith('{') or not pure_json_string.endswith('}'):
            raise ValueError("Sanitized string does not represent a valid JSON object.")

        try:
            print(f"🧪 Cleaned JSON string:\n{pure_json_string[:200]}...")

            data = json.loads(pure_json_string)

            # Ensure all required fields exist
            for key in ['slug', 'title_hash', 'keywords', 'summary', 'content']:
                if key not in data:
                    raise ValueError(f"Generated JSON missing essential key: {key}")

            return data

        except json.JSONDecodeError as e:
            error_detail = f"JSONDecodeError while parsing AI content: {e}"
            print(f"❌ {error_detail}")
            print(f"DEBUG: Problematic raw API response string that caused JSONDecodeError: '{pure_json_string[:500]}...'")
            log_ai_error(local_db_config, title, ai_provider_name, model_name_for_logging, None, error_detail, input_prompt_for_logging, generated_text)
            return fallback_generate_fields(title)

        except ValueError as e:
            error_detail = f"Parsing logic error: {e}"
            print(f"❌ {error_detail}")
            print(f"DEBUG: Problematic raw API response (ValueError context): '{generated_text[:500]}...'")
            log_ai_error(local_db_config, title, ai_provider_name, model_name_for_logging, None, error_detail, input_prompt_for_logging, generated_text)
            return fallback_generate_fields(title)

        except Exception as e:
            error_detail = f"Unexpected error during content parsing: {e}"
            print(f"❌ {error_detail}")
            print(f"DEBUG: Problematic raw API response (General Exception): '{generated_text[:500]}...'")
            log_ai_error(local_db_config, title, ai_provider_name, model_name_for_logging, None, error_detail, input_prompt_for_logging, generated_text)
            return fallback_generate_fields(title)

    except Exception as e:
        error_detail = f"Initial AI response structure or provider error: {e}"
        print(f"❌ {error_detail}")
        print(f"DEBUG: Raw AI response (before parsing): {api_response_json}")
        log_ai_error(local_db_config, title, ai_provider_name, model_name_for_logging, None, error_detail, input_prompt_for_logging, str(api_response_json))
        return fallback_generate_fields(title)


def call_ai_api_for_content(title, api_key, model_name, prompts_list, ai_provider_name, local_db_config):
    """
    Calls the AI API for content generation (text).
    Supports Together AI and AIML API (OpenAI-compatible).
    """
    url = None
    payload = {}
    input_prompt = None  # ✅ Prevents unbound local variable errors
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    normalized_provider = ai_provider_name.lower()

    if normalized_provider == 'together_ai':
        url = 'https://api.together.ai/v1/completions'
        prompt_elements = [
            "You are an assistant that writes blog post metadata.",
            f"Given the title \"{title}\", generate a JSON response with the following fields:",
            "- slug",
            "- title_hash (MD5 of the title)",
            "- keywords (comma-separated)",
            "- summary (short summary)",
            "- content (HTML paragraph)"
        ]
        if prompts_list:
            prompt_elements.append("\nHere are additional instructions/prompts to follow:")
            for custom_prompt in prompts_list:
                prompt_elements.append(f"- {custom_prompt}")
        prompt_elements.append("\nRespond only in raw JSON like this:")
        prompt_elements.append("""{
    "slug": "...",
    "title_hash": "...",
    "keywords": "...",
    "summary": "...",
    "content": "..."
}""")
        input_prompt = "\n".join(prompt_elements)  # ✅ Assigned here
        payload = {
            "model": model_name,
            "prompt": input_prompt,
            "max_tokens": 8192,
            "temperature": 0.7
        }

    elif normalized_provider == 'aiml_api':
        url = 'https://api.aimlapi.com/v1/chat/completions'
        prompt_elements = [
            "You are an assistant that writes blog post metadata.",
            f"Given the title \"{title}\", generate a comprehensive blog post content.",
            "Include a short summary, relevant keywords (comma-separated), a URL-friendly slug, and an MD5 hash of the title.",
            "Format the entire response as a JSON object with keys: 'slug', 'title_hash', 'keywords', 'summary', 'content' (HTML paragraph).",
            "Ensure the HTML content is well-formatted with <p> tags for paragraphs and of 1000 words."
        ]
        if prompts_list:
            prompt_elements.append("\nHere are additional instructions/prompts to follow:")
            for custom_prompt in prompts_list:
                prompt_elements.append(f"- {custom_prompt}")
        input_prompt = "\n".join(prompt_elements)  # ✅ Assigned here
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": input_prompt}],
            "max_tokens": 4096,
            "temperature": 0.7
        }

    else:
        error_msg = f"Unsupported content AI provider: {ai_provider_name}"
        log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, "N/A - Provider not supported")
        raise Exception(error_msg)

    if not api_key:
        error_msg = f"{ai_provider_name.upper().replace('_AI', '')}_API_KEY environment variable not set."
        log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, "N/A - API Key missing")
        raise Exception(error_msg)

    # Retry Logic
    max_retries = 5
    initial_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            raw_api_response_json = response.json()

            return safe_parse_json_response(title, raw_api_response_json, normalized_provider, input_prompt, local_db_config, model_name)

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            response_text = http_err.response.text
            if status_code == 429:
                print(f"Rate limit hit for '{title}' ({ai_provider_name}) (Attempt {attempt + 1}/{max_retries}). Retrying...")
                retry_after = http_err.response.headers.get('Retry-After')
                delay = int(retry_after) if retry_after else initial_delay * (2 ** attempt)
                time.sleep(delay)
                log_ai_error(local_db_config, title, ai_provider_name, model_name, status_code, "Rate limit hit. Retrying.", input_prompt, response_text, attempt + 1)
                continue
            else:
                error_msg = f"{ai_provider_name} Content API HTTP error {status_code}: {response_text}"
                log_ai_error(local_db_config, title, ai_provider_name, model_name, status_code, error_msg, input_prompt, response_text, attempt + 1)
                raise Exception(error_msg)

        except requests.exceptions.ConnectionError as conn_err:
            error_msg = f"Connection error for '{title}' ({ai_provider_name}): {conn_err}"
            log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, str(conn_err), attempt + 1)
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except requests.exceptions.Timeout as timeout_err:
            error_msg = f"Timeout error for '{title}' ({ai_provider_name}): {timeout_err}"
            log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, str(timeout_err), attempt + 1)
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except json.JSONDecodeError as json_err:
            error_msg = f"JSON decode error from {ai_provider_name} for '{title}': {json_err}. Response: {response.text[:300]}..."
            log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, response.text, attempt + 1)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error from {ai_provider_name} for '{title}': {e}"
            log_ai_error(local_db_config, title, ai_provider_name, model_name, None, error_msg, input_prompt, str(e), attempt + 1)
            raise Exception(error_msg)

    # Final failure
    final_error_msg = f"Failed to get {ai_provider_name} content for '{title}' after {max_retries} attempts."
    log_ai_error(local_db_config, title, ai_provider_name, model_name, None, final_error_msg, input_prompt, "N/A - Max retries exceeded", max_retries)
    raise Exception(final_error_msg)



def call_ai_api_for_image(prompt_text, api_key, model_name, ai_provider_name, local_db_config):
    """
    Calls the AI API for image generation, downloads the image locally to /static/uploads,
    and returns the local file path to store in the database.
    """
    url = None
    payload = {}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    normalized_provider = ai_provider_name.lower()

    if normalized_provider == 'together_ai':
        url = 'https://api.together.ai/v1/images/generations'
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "n": 1,
            "size": "1024x1024"
        }

    elif normalized_provider == 'aiml_api':
        url = 'https://api.aimlapi.com/v1/images/generations/'
        payload = {
            "model": model_name,
            "prompt": prompt_text
        }

    else:
        error_msg = f"Unsupported image AI provider: {ai_provider_name}"
        log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None,
                     error_msg, prompt_text, "N/A - Provider not supported")
        raise Exception(error_msg)

    if not api_key:
        error_msg = f"{ai_provider_name.upper().replace('_AI', '')}_API_KEY environment variable not set for image generation."
        log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None, error_msg, prompt_text, "N/A - API Key missing", log_type='image')

        raise Exception(error_msg)

    max_retries = 5
    initial_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code not in [200, 201]:
                raise requests.HTTPError(f"Unexpected status code: {response.status_code}", response=response)

            image_data = response.json()
            image_url = None

            if 'images' in image_data and isinstance(image_data['images'], list) and image_data['images']:
                image_url = image_data['images'][0].get('url')
            elif 'data' in image_data and isinstance(image_data['data'], list) and image_data['data']:
                image_url = image_data['data'][0].get('url')

            if image_url:
                # Download image to static/uploads and return local path
                local_path = download_image_to_static(image_url, prompt_text)
                return local_path
            else:
                error_msg = f"{ai_provider_name} image response missing or invalid format. Raw: {image_data}"
                log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, response.status_code,
                             error_msg, prompt_text, str(image_data), attempt + 1)
                raise Exception(error_msg)

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            response_text = http_err.response.text
            if status_code == 429:
                print(f"Rate limit hit for image generation (Attempt {attempt + 1}/{max_retries}). Retrying...")
                retry_after = http_err.response.headers.get('Retry-After')
                delay = int(retry_after) if retry_after else initial_delay * (2 ** attempt)
                time.sleep(delay)
                log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, status_code,
                             "Rate limit hit. Retrying.", prompt_text, response_text, attempt + 1)
                continue
            else:
                error_msg = f"{ai_provider_name} Image API HTTP error {status_code}: {response_text}"
                log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, status_code,
                             error_msg, prompt_text, response_text, attempt + 1)
                raise Exception(error_msg)

        except requests.exceptions.ConnectionError as conn_err:
            error_msg = f"Connection error for image generation ({ai_provider_name}): {conn_err}"
            print(f"{error_msg}. Retrying...")
            log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None,
                         error_msg, prompt_text, str(conn_err), attempt + 1)
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except requests.exceptions.Timeout as timeout_err:
            error_msg = f"Timeout error for image generation ({ai_provider_name}): {timeout_err}"
            print(f"{error_msg}. Retrying...")
            log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None,
                         error_msg, prompt_text, str(timeout_err), attempt + 1)
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except Exception as e:
            error_msg = f"Unexpected error during {ai_provider_name} image call: {e}"
            log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None,
                         error_msg, prompt_text, str(e), attempt + 1)
            raise Exception(error_msg)

    final_error_msg = f"Failed to get {ai_provider_name} image response after {max_retries} attempts."
    log_ai_error(local_db_config, prompt_text[:100], ai_provider_name, model_name, None,
                 final_error_msg, prompt_text, "N/A - Max retries exceeded", max_retries)
    raise Exception(final_error_msg)

# Add this helper function anywhere in utils.py
def download_image_to_static(image_url, title, upload_dir="static/uploads"):
    """
    Downloads an image from a URL and saves it locally under static/uploads.
    Returns the relative path (e.g., static/uploads/ferrari-abc123.png).
    """
    try:
        os.makedirs(upload_dir, exist_ok=True)

        ext = os.path.splitext(urlparse(image_url).path)[1] or ".png"
        safe_title = slugify(title)[:70]  # limit filename slug to avoid Windows path limit
        unique_suffix = uuid4().hex[:8]
        filename = f"{safe_title}-{unique_suffix}{ext}"
        local_path = os.path.join(upload_dir, filename)

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Image saved to: {local_path}")
            return local_path.replace("\\", "/")
        else:
            raise Exception(f"Failed to download image, status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to download and save image: {e}")
        return None

