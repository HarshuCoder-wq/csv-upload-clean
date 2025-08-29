import os
import logging
import requests
import base64
import mimetypes
import io
import openai

# Enable logging
logging.basicConfig(level=logging.INFO)

# API Keys (ensure they are set in your environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
AIML_API_KEY = os.getenv("AIML_API_KEY")

openai.api_key = OPENAI_API_KEY


def get_image_mime_type(file):
    mime_type, _ = mimetypes.guess_type(file.name if hasattr(file, 'name') else None)
    logging.info(f"Detected MIME type: {mime_type}")
    return mime_type or "image/jpeg"


def encode_image_to_base64(image_file):
    image_file.seek(0)
    encoded = base64.b64encode(image_file.read()).decode("utf-8")
    mime_type = get_image_mime_type(image_file)
    image_file.seek(0)  # Reset stream after reading
    return f"data:{mime_type};base64,{encoded}"


def duplicate_image_file(image_file):
    image_file.seek(0)
    return io.BytesIO(image_file.read())


def call_openai_api(prompt, image_url=None):
    messages = []

    if image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt
        })

    try:
        logging.info("Calling OpenAI Chat Completions with vision model...")
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )
        result = response.choices[0].message.content
        logging.info(f"OpenAI Response: {result}")
        return result
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise


def call_together_qwen_vl(prompt, image_file=None):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    messages = []

    if image_file:
        image_data_url = encode_image_to_base64(image_file)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "messages": messages,
        "max_tokens": 10000
    }

    response = requests.post(url, json=payload, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_comet_ai(prompt):
    url = "https://api.comet.com/v1/generate"
    headers = {"Authorization": f"Bearer {COMET_API_KEY}"}
    payload = {
        "model": "comet-model-name",
        "prompt": prompt,
        "max_tokens": 8192
    }

    response = requests.post(url, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json().get("output", "")


def call_aiml_api(prompt, image_url=None):
    url = "https://api.aimlapi.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {AIML_API_KEY}"}

    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8000
    }

    response = requests.post(url, json=payload, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")


def looks_like_image_was_ignored(response_text):
    fail_phrases = [
        "i can't view or interpret images",
        "please describe the image",
        "i'm sorry, i can't see",
        "i don't have the ability to see images",
        "unable to process the image",
        "no image was provided",
        "i can't analyze images",
        "i can't view images"
    ]
    lower_text = response_text.strip().lower()
    return not lower_text or any(phrase in lower_text for phrase in fail_phrases)


def generate_content_from_prompt(prompt, image_file=None, image_url=None):
    """
    Main function tries OpenAI API first with image_url,
    then fallbacks to Together Qwen VL (with image_file),
    AIML (with base64 image),
    and finally Comet (text only).
    """
    errors = []

    if image_file:
        base64_image = encode_image_to_base64(image_file)
        # Duplicate image file streams for multiple API calls
        image_file1 = duplicate_image_file(image_file)
        image_file2 = duplicate_image_file(image_file)
    else:
        base64_image = None
        image_file1 = image_file2 = None

    # 0. Try OpenAI API first (requires image_url, not base64)
    try:
        if not image_url and base64_image:
            logging.warning("OpenAI API requires public image URL; base64 will not work.")
        logging.info("Calling OpenAI API first...")
        result = call_openai_api(prompt, image_url=image_url)
        if not looks_like_image_was_ignored(result):
            return f"[OpenAI GPT-4o] {result}"
        logging.warning("OpenAI ignored the image or no good response. Trying fallback...")
        errors.append("OpenAI said it can't interpret the image.")
    except Exception as e:
        logging.error(f"OpenAI API failed: {e}")
        errors.append(f"call_openai_api failed: {str(e)}")

    # 1. Try Together Qwen VL (vision-capable, base64 image)
    try:
        logging.info("Calling Together Qwen VL API...")
        result = call_together_qwen_vl(prompt, image_file=image_file2)
        if not looks_like_image_was_ignored(result):
            return f"[Together Qwen VL] {result}"
        logging.warning("Together Qwen VL ignored the image. Trying next fallback...")
        errors.append("Together Qwen VL said it can't interpret the image.")
    except Exception as e:
        logging.error(f"Together Qwen VL API failed: {e}")
        errors.append(f"call_together_qwen_vl failed: {str(e)}")

    # 2. Try AIML (GPT-4o) - base64 image
    try:
        logging.info("Calling AIML API...")
        result = call_aiml_api(prompt, image_url=base64_image)
        if not looks_like_image_was_ignored(result):
            return f"[AIML GPT-4o] {result}"
        logging.warning("AIML ignored the image. Trying fallback...")
        errors.append("AIML said it can't interpret the image.")
    except Exception as e:
        logging.error(f"AIML API failed: {e}")
        errors.append(f"call_aiml_api failed: {str(e)}")

    # 3. Fallback to Comet (text only)
    try:
        logging.info("Calling Comet AI (text only fallback)...")
        result = call_comet_ai(prompt)
        return f"[Comet AI Fallback] {result}"
    except Exception as e:
        logging.error(f"Comet AI failed: {e}")
        errors.append(f"call_comet_ai failed: {str(e)}")

    # All failed
    raise Exception("All AI APIs failed.\n" + "\n".join(errors))


# Example usage
if __name__ == "__main__":
    prompt_text = "Describe the image content."
    test_image_url = "https://yourdomain.com/static/uploads/example_image.webp"  # public URL to image

    # You can open the image file here and pass it as `image_file` if needed
    image_file = None  # or open("path_to_image", "rb")

    try:
        answer = generate_content_from_prompt(prompt_text, image_file=image_file, image_url=test_image_url)
        print("Answer:\n", answer)
    except Exception as err:
        print("Error:", err)
