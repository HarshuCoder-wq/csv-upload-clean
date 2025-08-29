import os
import csv
import mysql.connector # Assuming you are using mysql.connector
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib
import requests
import json
import time


csv.field_size_limit(30000000)

# Define your INSERT_QUERY here. This has been updated to match the post_data_tuple.
INSERT_QUERY = """
    INSERT INTO posts (
        lang_id, title, slug, title_hash, keywords, summary, content,
        optional_url, pageviews, comment_count, need_auth, slider_order,
        featured_order, is_scheduled, visibility, show_right_column,
        post_type, video_path, video_storage, image_url, video_url,
        video_embed_code, status, feed_id, post_url, show_post_url,
        image_description, show_item_numbers, is_poll_public,
        link_list_style, recipe_info, post_data, category_id, user_id,
        created_at, updated_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s
    )
"""

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

def safe_parse_json_response(title, api_response_json, ai_provider_name):
    """
    Parses the raw JSON response object from an AI API.
    Handles specific structures for Together AI/Comet vs. AIML API
    and attempts to extract the final JSON content, including stripping markdown fences.
    """
    try:
        normalized_provider = ai_provider_name.lower()
        generated_text = ""

        if normalized_provider == 'aiml_api':
            if 'choices' in api_response_json and api_response_json['choices'] and \
               'message' in api_response_json['choices'][0] and \
               'content' in api_response_json['choices'][0]['message']:
                generated_text = api_response_json['choices'][0]['message']['content']
            else:
                raise ValueError("AIML API response missing expected 'choices[0].message.content' structure.")

        elif normalized_provider in ['together_ai', 'comet']:
            generated_text = api_response_json.get('choices', [{}])[0].get('text', '')
            if not generated_text:
                raise ValueError("Together AI/Comet response missing 'text' field in choices.")
        else:
            raise ValueError(f"Unsupported AI provider for content parsing: {ai_provider_name}")

        # --- Robust JSON extraction logic ---
        # Try to find the outermost JSON object by finding the first '{' and last '}'
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1

        if json_start == -1 or json_end <= json_start:
            # If no valid JSON structure is found, try to strip markdown fences and re-check
            cleaned_text = generated_text.replace('```json', '').replace('```', '').strip()
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                print(f"DEBUG: No clear JSON object found in AI content for '{title}' from {ai_provider_name}. Raw text: {generated_text[:500]}...")
                raise ValueError("No valid JSON object found in AI response text after initial and markdown stripping attempts.")
            pure_json_string = cleaned_text[json_start:json_end]
        else:
            pure_json_string = generated_text[json_start:json_end]
        # --- End robust JSON extraction logic ---


        try:
            data = json.loads(pure_json_string)

            # Validate essential keys for all providers
            for key in ['slug', 'title_hash', 'keywords', 'summary', 'content']:
                if key not in data:
                    raise ValueError(f"Generated JSON missing essential key: {key}")
            return data

        except json.JSONDecodeError as e:
            print(f"❌ JSONDecodeError while parsing AI content for '{title}' from {ai_provider_name}: {e}")
            print(f"DEBUG: Problematic raw API response string that caused JSONDecodeError: '{pure_json_string[:500]}...'")
            return fallback_generate_fields(title)
        except ValueError as e:
            print(f"❌ Parsing logic error for '{title}' from {ai_provider_name}: {e}")
            print(f"DEBUG: Problematic raw API response (ValueError context - generated_text): '{generated_text[:500]}...'")
            return fallback_generate_fields(title)
        except Exception as e:
            print(f"❌ Unexpected error during content parsing for '{title}' from {ai_provider_name}: {e}")
            print(f"DEBUG: Problematic raw API response (General Exception context - generated_text): '{generated_text[:500]}...'")
            return fallback_generate_fields(title)

    except Exception as e:
        # Catch any errors from the initial structure extraction or provider check
        print(f"❌ Initial AI response structure or provider error for '{title}' from {ai_provider_name}: {e}")
        print(f"DEBUG: Raw AI response (before any parsing attempts): {api_response_json}")
        return fallback_generate_fields(title)


def call_ai_api_for_content(title, api_key, model_name, prompts_list, ai_provider_name):
    """
    Calls the AI API for content generation (text).
    Supports Together AI and AIML API (OpenAI-compatible).
    """
    url = None
    payload = {}
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
            "- content (HTML paragraph 1000 words)"
        ]

        if prompts_list:
            prompt_elements.append("\nAdditional instructions to follow:")
            for custom_prompt in prompts_list:
                prompt_elements.append(f"- {custom_prompt}")

        # Strong formatting guidance
        prompt_elements.append("\n⚠️ Important Formatting Instructions:")
        prompt_elements.append("- Respond ONLY with raw JSON.")
        prompt_elements.append("- Do NOT use code blocks, markdown, triple backticks, or any extra explanation.")
        prompt_elements.append("- Start the response with '{' and end it with '}'.")
        prompt_elements.append("- Your entire output MUST be a valid JSON object.")

        # Example JSON template
        prompt_elements.append("\nExpected JSON format:")
        prompt_elements.append("""{
  "slug": "...",
  "title_hash": "...",
  "keywords": "...",
  "summary": "...",
  "content": "..."
}""")

        prompt = "\n".join(prompt_elements)

        payload = {
            "model": model_name,
            "prompt": prompt,
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
            "Ensure the HTML content is well-formatted with <p> tags for paragraphs amd in 1000 words.",
            "Respond ONLY with raw JSON. No markdown, no explanations, no code blocks. Begin with '{' and end with '}'."
        ]

        if prompts_list:
            prompt_elements.append("\nAdditional instructions to follow:")
            for custom_prompt in prompts_list:
                prompt_elements.append(f"- {custom_prompt}")

        prompt = "\n".join(prompt_elements)

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.7
        }

    else:
        raise Exception(f"Unsupported content AI provider: {ai_provider_name}")

    if not api_key:
        raise Exception(f"{ai_provider_name.upper().replace('_AI', '')}_API_KEY environment variable not set.")

    max_retries = 5
    initial_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            raw_api_response_json = response.json()
            parsed_data = safe_parse_json_response(title, raw_api_response_json, normalized_provider)

            # Optionally retry with another model if fallback content was returned
            if "fallback content" in parsed_data.get("content", "").lower():
                raise Exception("Fallback triggered due to invalid AI JSON.")

            return parsed_data

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"Rate limit hit for content '{title}' ({ai_provider_name}) (Attempt {attempt + 1}/{max_retries}). Retrying after delay...")
                retry_after = http_err.response.headers.get('Retry-After')
                delay = int(retry_after) if retry_after else initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                print(f"HTTP Error Response Text: {http_err.response.text}")
                raise Exception(f"{ai_provider_name} Content API HTTP error {http_err.response.status_code}: {http_err.response.text}")

        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error for content '{title}' ({ai_provider_name}) (Attempt {attempt + 1}/{max_retries}): {conn_err}. Retrying...")
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error for content '{title}' ({ai_provider_name}) (Attempt {attempt + 1}/{max_retries}): {timeout_err}. Retrying...")
            time.sleep(initial_delay * (2 ** attempt))
            continue

        except json.JSONDecodeError as json_err:
            print(f"Failed to decode initial JSON response from {ai_provider_name} API for '{title}': {json_err}")
            raise Exception(f"Invalid JSON response from {ai_provider_name} API for '{title}'")

        except Exception as e:
            print(f"⚠️ Exception during attempt {attempt + 1} for '{title}': {e}")
            continue

    raise Exception(f"Failed to get {ai_provider_name} content response for '{title}' after {max_retries} attempts.")



def call_ai_api_for_image(prompt_text, api_key, model_name, ai_provider_name):
    """
    Calls the AI API for image generation.
    Currently supports Together AI.
    """
    url = None
    payload = {}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    normalized_provider = ai_provider_name.lower()

    if normalized_provider == 'together_ai':
        url = 'https://api.together.ai/v1/images/generations'
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "n": 1, 
            "size": "800x800" # Example size, adjust as needed
        }
    else:
        raise Exception(f"Unsupported image AI provider: {ai_provider_name}")

    if not api_key:
        raise Exception(f"{ai_provider_name.upper().replace('_AI', '')}_API_KEY environment variable not set for image generation.")

    max_retries = 5
    initial_delay = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60) # Increased timeout
            response.raise_for_status()

            image_data = response.json()

            if 'data' in image_data and image_data['data']:
                image_url = image_data['data'][0].get('url')
                if image_url:
                    return image_url
                else:
                    raise Exception(f"{ai_provider_name} image response 'data' field missing 'url'.")
            else:
                raise Exception(f"{ai_provider_name} image response missing 'data' field or is empty.")
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                print(f"Rate limit hit for image generation (Attempt {attempt + 1}/{max_retries}). Retrying after delay...")
                retry_after = http_err.response.headers.get('Retry-After')
                delay = int(retry_after) if retry_after else initial_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                raise Exception(f"{ai_provider_name} Image API HTTP error {http_err.response.status_code}: {http_err.response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error for image generation (Attempt {attempt + 1}/{max_retries}): {conn_err}. Retrying...")
            time.sleep(initial_delay * (2 ** attempt))
            continue
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error for image generation (Attempt {attempt + 1}/{max_retries}): {timeout_err}. Retrying...")
            time.sleep(initial_delay * (2 ** attempt))
            continue
        except Exception as e:
            raise Exception(f"An unexpected error occurred during {ai_provider_name} image call: {e}")

    raise Exception(f"Failed to get {ai_provider_name} image response after {max_retries} attempts.")

def process_single_row(row_data, content_ai_config, image_ai_config, prompts_list, selected_category_id, selected_user_id):
    """
    Processes a single row to generate AI content and potentially an image.
    This function will be run in a thread.
    """
    title = row_data.get('title', '').strip()
    if not title:
        return None, None # Return None for row and image URL if title is empty

    lang_id = int(row_data.get('lang_id', 0)) if row_data.get('lang_id') else 1

    ai_data = None
    generated_image_url = ""

    # --- Content Generation with Multi-Model Fallback ---
    # Iterate through each provider
    for provider_name, api_key_env_var, models_list in content_ai_config['providers']:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            print(f"❗ Skipping content generation with {provider_name} due to missing {api_key_env_var} environment variable for '{title}'.")
            continue

        # Iterate through models for the current provider
        for model_name in models_list:
            try:
                print(f"Attempting content generation for '{title}' with {provider_name} using model: {model_name}...")
                ai_data = call_ai_api_for_content(title, api_key, model_name, prompts_list, provider_name)
                print(f"✅ Content generated for '{title}' using {model_name} via {provider_name}")
                break # Success, break from model loop
            except Exception as e:
                print(f"⚠️ Content AI generation failed for '{title}' with {provider_name} ({model_name}): {e}. Trying next model for this provider.")
                ai_data = None # Reset ai_data for next attempt

        if ai_data: # If content was successfully generated by any model of the current provider
            break # Break from provider loop as well

    if not ai_data:
        print(f"❌ All content AI models failed for '{title}'. Using fallback fields.")
        ai_data = fallback_generate_fields(title)

    # --- Image Generation (Optional, with Multi-Model Fallback if multiple are passed) ---
    if image_ai_config.get('enabled', False):
        # Iterate through image providers and their models (though current config has only one model per provider)
        for provider_name, api_key_env_var, model_name in image_ai_config['providers']:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                print(f"❗ Skipping image generation with {provider_name} due to missing {api_key_env_var} environment variable for '{title}'.")
                continue

            # If model_name is a list (for future multi-image-model support), iterate. Otherwise, treat as single.
            current_image_models = [model_name] if not isinstance(model_name, list) else model_name

            for img_model_name in current_image_models:
                image_prompt = f"A high-quality, professional image for a blog post titled: '{title}'. Keywords: {ai_data.get('keywords', '')}. Summary: {ai_data.get('summary', '')}"
                try:
                    print(f"Attempting image generation for '{title}' with {provider_name} using model: {img_model_name}...")
                    generated_image_url = call_ai_api_for_image(image_prompt, api_key, img_model_name, provider_name)
                    print(f"🖼️ Image generated for '{title}' using {img_model_name} via {provider_name}: {generated_image_url}")
                    break # Success, break from image model loop
                except Exception as e:
                    print(f"⚠️ Image AI generation failed for '{title}' with {provider_name} ({img_model_name}): {e}. Trying next image model.")
                    generated_image_url = "" # Reset for next attempt

            if generated_image_url: # If image was successfully generated by any model of the current provider
                break # Break from image provider loop as well
    else:
        print("❗ Image generation skipped as 'image_ai_config['enabled']' was False.")
        generated_image_url = "" # Ensure it's empty if not enabled


    final_category_id = int(selected_category_id) if selected_category_id else None
    final_user_id = int(selected_user_id) if selected_user_id else None
    now = datetime.now()

    post_data_tuple = (
        lang_id, title, ai_data.get('slug'), ai_data.get('title_hash'),
        ai_data.get('keywords'), ai_data.get('summary'), ai_data.get('content'),
        '', # optional_url
        None, # pageviews
        0, # comment_count
        0, # need_auth
        0, # slider_order
        0, # featured_order
        0, # is_scheduled
        1, # visibility (public)
        0, # show_right_column
        'article', # post_type
        '', # video_path
        '', # video_storage
        generated_image_url, # image_url
        '', # video_url
        '', # video_embed_code
        1, # status (published)
        None, # feed_id
        '', # post_url
        0, # show_post_url
        '', # image_description
        0, # show_item_numbers
        0, # is_poll_public
        '', # link_list_style
        '', # recipe_info
        '', # post_data
        final_category_id,
        final_user_id,
        now,
        now
    )
    return post_data_tuple, generated_image_url


def upload_csv_with_file(csv_path,
    db_config,
    content_ai_config,
    image_ai_config,
    prompts_list,
    local_db_config, # New parameter for local DB config
    selected_category_id=None,
    selected_user_id=None,
    selected_domain_id=None):

    conn = None # Initialize conn for live DB to None
    local_conn = None # Initialize conn for local DB to None
    generated_image_urls_for_display = [] # To collect image URLs for result.html
    csv_file_name = os.path.basename(csv_path)
    created_at = datetime.now()

    # Initialize counters *before* any processing starts
    total_rows_processed = 0 # Will count successfully processed and inserted rows
    all_rows = []            # Will store all rows read from the CSV

    try:
        # --- Live Database Connection Establishment ---
        cleaned_db_config = db_config.copy() if db_config is not None else {}
        unsupported_keys = ['connection', 'label']
        for key in unsupported_keys:
            if key in cleaned_db_config:
                del cleaned_db_config[key]

        if not isinstance(cleaned_db_config, dict):
            raise ValueError(f"Database configuration (db_config) must be a dictionary, but received type: {type(db_config)}")
        if not cleaned_db_config:
            raise ValueError("Database configuration (db_config) is empty or None. Please ensure it contains valid connection parameters.")

        conn = mysql.connector.connect(**cleaned_db_config)
        print("✅ Live database connection established.")

        # --- Local Database Connection Establishment for ai_campaigns ---
        cleaned_local_db_config = local_db_config.copy() if local_db_config is not None else {}
        for key in unsupported_keys: # Use same unsupported keys for local config cleanup
            if key in cleaned_local_db_config:
                del cleaned_local_db_config[key]

        if not isinstance(cleaned_local_db_config, dict):
            raise ValueError(f"Local database configuration (local_db_config) must be a dictionary, but received type: {type(local_db_config)}")
        if not cleaned_local_db_config:
            raise ValueError("Local database configuration (local_db_config) is empty or None. Please ensure it contains valid connection parameters.")

        local_conn = mysql.connector.connect(**cleaned_local_db_config)
        print("✅ Local database connection established for ai_campaigns.")

        # Ensure ai_campaigns table exists in the local database
        with local_conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_campaigns (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    csv_file_name VARCHAR(255) NOT NULL,
                    total_titles INT NOT NULL,
                    success_count INT NOT NULL,
                    fail_count INT NOT NULL,
                    created_at DATETIME NOT NULL,
                    db_host_key VARCHAR(255)
                )
            """)
            local_conn.commit()
            print("🗄️ Ensured 'ai_campaigns' table exists in local database.")


        # --- Read CSV File (Moved here, after connections) ---
        with open(csv_path, 'r', encoding='latin1') as f:
            reader = csv.DictReader(f)
            for r in reader:
                all_rows.append(r)

        num_titles = len(all_rows)
        print(f"Total titles to process: {num_titles}")

        num_parts = 1
        if num_titles >= 50:
            num_parts = 3
            print(f"Dividing {num_titles} titles into {num_parts} parts for processing.")

        rows_per_part = (num_titles + num_parts - 1) // num_parts
        max_threads = 5

        # --- Main Processing Loop ---
        for part_idx in range(num_parts):
            start_idx = part_idx * rows_per_part
            end_idx = min((part_idx + 1) * rows_per_part, num_titles)
            part_rows = all_rows[start_idx:end_idx]

            if not part_rows:
                continue

            print(f"\n--- Processing Part {part_idx + 1}/{num_parts} (Titles {start_idx + 1} to {end_idx}) ---")

            batch_for_db = []
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {executor.submit(process_single_row, row,
                                            content_ai_config,
                                            image_ai_config,
                                            prompts_list, selected_category_id, selected_user_id): row
                           for row in part_rows}

                for future in as_completed(futures):
                    original_row = futures[future]
                    title_for_debug = original_row.get('title', 'N/A')
                    try:
                        post_tuple, img_url = future.result()
                        if post_tuple:
                            batch_for_db.append(post_tuple)
                        if img_url:
                            generated_image_urls_for_display.append({'title': title_for_debug, 'image_url': img_url})

                    except Exception as exc:
                        print(f"❌ Error processing title '{title_for_debug}': {exc}")

                    batch_size = 5000
                    if len(batch_for_db) >= batch_size:
                        try:
                            with conn.cursor(buffered=True) as cursor:
                                cursor.executemany(INSERT_QUERY, batch_for_db)
                                conn.commit()
                                total_rows_processed += len(batch_for_db)
                                print(f"📦 Inserted {len(batch_for_db)} rows. Total inserted: {total_rows_processed}.")
                            batch_for_db.clear()
                        except Exception as db_exc:
                            print(f"❌ Error during batch DB insert: {db_exc}")
                            conn.rollback()
                            batch_for_db.clear()


            if batch_for_db:
                try:
                    with conn.cursor(buffered=True) as cursor:
                        cursor.executemany(INSERT_QUERY, batch_for_db)
                        conn.commit()
                        total_rows_processed += len(batch_for_db)
                        print(f"📦 Inserted {len(batch_for_db)} rows from Part {part_idx + 1}. Total inserted: {total_rows_processed}.")
                    batch_for_db.clear()
                except Exception as db_exc:
                    print(f"❌ Error during final batch DB insert for Part {part_idx + 1}: {db_exc}")
                    conn.rollback()
                    batch_for_db.clear()

        # --- Final Summary & Campaign Logging (Using local_conn) ---
        success_count = total_rows_processed
        fail_count = num_titles - success_count
        db_host_key = db_config.get('host') if isinstance(db_config, dict) else 'unknown' # This refers to the live DB host

        try:
            with local_conn.cursor() as cursor: # Use local_conn here
                cursor.execute("""
                    INSERT INTO ai_campaigns (csv_file_name, total_titles, success_count, fail_count, created_at, db_host_key)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    csv_file_name,
                    num_titles,
                    success_count,
                    fail_count,
                    created_at,
                    db_host_key
                ))
                local_conn.commit() # Commit on local_conn
                print("📊 AI Campaign summary saved to ai_campaigns.")
        except Exception as e:
            print(f"❌ Failed to save campaign summary: {e}")
            # Do NOT rollback local_conn here if live_conn failed, they are independent
            # local_conn.rollback() # Only rollback if the summary insert itself failed

        print("✅ CSV upload completed successfully.")
        return generated_image_urls_for_display

    except ValueError as ve:
        print(f"❌ Configuration Error: {ve}")
        if conn:
            conn.rollback()
        if local_conn:
            local_conn.rollback()
        raise

    except Exception as e:
        if conn:
            conn.rollback()
        if local_conn:
            local_conn.rollback()
        print(f"❌ Error during upload: {e}")
        raise
    finally:
        if conn:
            conn.close()
            print("🔗 Live database connection closed.")
        if local_conn:
            local_conn.close()
            print("🔗 Local database connection closed.")
