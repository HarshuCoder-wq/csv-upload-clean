import os
import html
import csv
import mysql.connector
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib
import requests
import json
import time

# Import helper functions from the new utils.py
from uploader.utils import slugify, fallback_generate_fields, safe_parse_json_response, \
                           call_ai_api_for_content, call_ai_api_for_image, log_ai_error, _xml_escape, chunk_list


csv.field_size_limit(30000000)

# INSERT_QUERY for the local_generated_posts table
LOCAL_POSTS_INSERT_QUERY = """
    INSERT INTO local_generated_posts (
        lang_id, title, slug, title_hash, keywords, summary, content,
        optional_url, pageviews, comment_count, need_auth, slider_order,
        featured_order, is_scheduled, visibility, show_right_column,
        post_type, video_path, video_storage, image_url, video_url,
        video_embed_code, status, feed_id, post_url, show_post_url,
        image_description, show_item_numbers, is_poll_public,
        link_list_style, recipe_info, post_data, category_id, user_id,
        domain_id, created_at, updated_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s
    )
"""

def generate_and_save_local_posts(csv_path,
                                  content_ai_config,
                                  image_ai_config,
                                  prompts_list,
                                  local_db_config,
                                  selected_category_id=None,
                                  selected_user_id=None,
                                  selected_domain_label=None,
                                  selected_domain_id=None):
    """
    Generates AI content and images for posts from a CSV, saves them to the
    local database's local_generated_posts table, and logs campaign summary.
    Returns a list of successfully generated post data for RSS feed creation.
    """
    local_conn = None
    generated_posts_for_rss = []
    csv_file_name = os.path.basename(csv_path)
    created_at = datetime.now()

    total_rows_processed = 0
    all_rows = []

    try:
        # --- DB Connection ---
        cleaned_local_db_config = local_db_config.copy() if local_db_config else {}
        for key in ['connection', 'label']:
            cleaned_local_db_config.pop(key, None)

        if not isinstance(cleaned_local_db_config, dict) or not cleaned_local_db_config:
            raise ValueError("Invalid local_db_config.")

        local_conn = mysql.connector.connect(**cleaned_local_db_config)
        print("✅ Connected to local DB")

        with local_conn.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS local_generated_posts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                lang_id INT,
                title VARCHAR(255) NOT NULL,
                slug VARCHAR(255),
                title_hash VARCHAR(32),
                keywords TEXT,
                summary TEXT,
                content LONGTEXT,
                optional_url VARCHAR(255),
                pageviews INT DEFAULT 0,
                comment_count INT DEFAULT 0,
                need_auth TINYINT(1) DEFAULT 0,
                slider_order INT DEFAULT 0,
                featured_order INT DEFAULT 0,
                is_scheduled TINYINT(1) DEFAULT 0,
                visibility TINYINT(1) DEFAULT 1,
                show_right_column TINYINT(1) DEFAULT 0,
                post_type VARCHAR(50) DEFAULT 'article',
                video_path VARCHAR(255),
                video_storage VARCHAR(50),
                image_url VARCHAR(255),
                video_url VARCHAR(255),
                video_embed_code TEXT,
                status TINYINT(1) DEFAULT 1,
                feed_id INT,
                post_url VARCHAR(255),
                show_post_url TINYINT(1) DEFAULT 0,
                image_description TEXT,
                show_item_numbers TINYINT(1) DEFAULT 0,
                is_poll_public TINYINT(1) DEFAULT 0,
                link_list_style VARCHAR(50),
                recipe_info TEXT,
                post_data TEXT,
                category_id INT,
                user_id INT,
                domain_id INT(11),
                created_at DATETIME,
                updated_at DATETIME
            )""")

            cursor.execute("""CREATE TABLE IF NOT EXISTS blog_ai_error_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                blog_title VARCHAR(255),
                provider_name VARCHAR(100),
                model_name VARCHAR(100),
                status_code INT,
                error_message TEXT,
                input_prompt TEXT,
                response TEXT,
                retry_attempt INT,
                created_at DATETIME NOT NULL
            )""")
            local_conn.commit()

        # --- Read CSV ---
        with open(csv_path, 'r', encoding='latin1') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        num_titles = len(all_rows)
        print(f"📄 Titles to process: {num_titles}")

        # --- Batch Processing Logic ---
        batch_size = 5 # Set your desired batch size here
        
        # Get provider lists from configuration
        image_providers = image_ai_config.get('providers', [])
        content_providers = content_ai_config.get('providers', [])

        if len(image_providers) < 2 or len(content_providers) < 2:
            print("⚠️ Not enough AI providers configured for batch processing. Processing all posts with the default fallback logic.")
            # If not enough providers, process all rows with the default config
            rows_to_process = chunk_list(all_rows, len(all_rows)) # A single chunk
            
        else:
            print(f"⚙️ Processing in batches of {batch_size}, alternating between providers...")
            rows_to_process = chunk_list(all_rows, batch_size)

        post_count = 0
        for i, row_chunk in enumerate(rows_to_process):
            # Determine which content provider's configuration to use for this batch
            if i % 2 == 0:
                # Use the first content provider for even-indexed chunks
                current_content_config = {'providers': [content_providers[0]]}
                print(f"\n--- Processing batch {i+1} with content provider: {content_providers[0][0]} ---")
            else:
                # Use the second content provider for odd-indexed chunks
                current_content_config = {'providers': [content_providers[1]]}
                print(f"\n--- Processing batch {i+1} with content provider: {content_providers[1][0]} ---")

            # ✅ Pass all image providers for full fallback (DO NOT limit to batch)
            current_image_config = image_ai_config

            # Process the current chunk using a ThreadPoolExecutor
            batch_for_db = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(process_single_row_local, row,
                                    current_content_config, current_image_config, prompts_list,
                                    selected_category_id, selected_user_id,
                                    selected_domain_id, local_db_config): row
                    for row in row_chunk
                }

                for future in as_completed(futures):
                    original_row = futures[future]
                    title_for_debug = original_row.get('title', 'N/A')
                    post_count += 1
                    print(f"--- Processing post {post_count}: '{title_for_debug}' ---")
                    try:
                        post_tuple, img_url = future.result()
                        if post_tuple:
                            batch_for_db.append(post_tuple)
                            post_data = {
                                'title': post_tuple[1], 'summary': post_tuple[5],
                                'content': post_tuple[6], 'image_url': post_tuple[19],
                                'slug': post_tuple[2], 'created_at': post_tuple[35],
                                'category_id': post_tuple[33], 'domain_id': post_tuple[34]
                            }
                            # Get category_slug for this post
                            category_id_for_slug = post_data['category_id']
                            try:
                                with local_conn.cursor(dictionary=True) as slug_cursor:
                                    slug_cursor.execute("SELECT slug FROM categories WHERE id = %s", (category_id_for_slug,))
                                    row = slug_cursor.fetchone()
                                    post_data['category_slug'] = row['slug'] if row else 'uncategorized'
                            except Exception as e:
                                print(f"⚠️ Couldn't fetch category_slug for ID {category_id_for_slug}: {e}")
                                post_data['category_slug'] = 'uncategorized'

                            generated_posts_for_rss.append(post_data)

                    except Exception as exc:
                        print(f"❌ Error processing '{title_for_debug}': {exc}")
                        log_ai_error(local_db_config, title_for_debug, "N/A", "N/A", None, str(exc), "N/A", "N/A")

            # Insert batch into DB
            if batch_for_db:
                try:
                    with local_conn.cursor(buffered=True) as cursor:
                        cursor.executemany(LOCAL_POSTS_INSERT_QUERY, batch_for_db)
                        local_conn.commit()
                        total_rows_processed += len(batch_for_db)
                        print(f"✅ Inserted {len(batch_for_db)} rows.")
                except Exception as db_exc:
                    print(f"❌ DB Insert Error: {db_exc}")
                    local_conn.rollback()
                    log_ai_error(local_db_config, "Batch Insert", "MySQL", "System", None, str(db_exc), "N/A", str(batch_for_db))


            # Insert batch into DB
            if batch_for_db:
                try:
                    with local_conn.cursor(buffered=True) as cursor:
                        cursor.executemany(LOCAL_POSTS_INSERT_QUERY, batch_for_db)
                        local_conn.commit()
                        total_rows_processed += len(batch_for_db)
                        print(f"✅ Inserted {len(batch_for_db)} rows.")
                except Exception as db_exc:
                    print(f"❌ DB Insert Error: {db_exc}")
                    local_conn.rollback()
                    log_ai_error(local_db_config, "Batch Insert", "MySQL", "System", None, str(db_exc), "N/A", str(batch_for_db))

        # 📦 Final Debug
        print(f"🔍 Total posts collected for RSS: {len(generated_posts_for_rss)}")
        if generated_posts_for_rss:
            print(f"🧾 First post: {generated_posts_for_rss[0]}")
        else:
            print("⚠️ No posts were collected for RSS.")

        # Campaign summary log
        success_count = total_rows_processed
        fail_count = num_titles - success_count
        db_host_label = selected_domain_label or "Local"

        try:
            with local_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ai_campaigns (csv_file_name, total_titles, success_count, fail_count, created_at, db_host_key)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    csv_file_name, num_titles, success_count, fail_count,
                    created_at, db_host_label
                ))
                local_conn.commit()
                print("📊 Campaign summary saved.")
        except Exception as e:
            print(f"⚠️ Failed to save campaign log: {e}")
            log_ai_error(local_db_config, csv_file_name, "Campaign Summary", "MySQL", None, str(e), "N/A", "N/A")

        return generated_posts_for_rss

    except Exception as e:
        if local_conn:
            local_conn.rollback()
        print(f"❌ Unexpected Error: {e}")
        log_ai_error(local_db_config, csv_file_name, "General Error", "System", None, str(e), "N/A", "N/A")
        raise
    finally:
        if local_conn:
            local_conn.close()
            print("🔒 DB connection closed.")                               


def process_single_row_local(row_data, content_ai_config, image_ai_config, prompts_list, selected_category_id, selected_user_id, selected_domain_id, local_db_config):
    """
    Processes a single row for local saving, leveraging AI content/image generation.
    This function will be run in a thread. It skips saving fallback content if AI fails.
    """
    title = row_data.get('title', '').strip()
    if not title:
        return None, None  # Skip empty title

    lang_id = int(row_data.get('lang_id', 0)) if row_data.get('lang_id') else 1

    ai_data = None
    generated_image_url = ""

    # --- Content Generation with Multi-Model Fallback ---
    # The config now only contains the provider for this specific batch
    for provider_name, api_key_env_var, models_list in content_ai_config['providers']:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            print(f"❗ Skipping content generation with {provider_name} due to missing {api_key_env_var} environment variable for '{title}'.")
            continue

        for model_name in models_list:
            try:
                print(f"Attempting content generation for '{title}' with {provider_name} using model: {model_name}...")
                ai_data = call_ai_api_for_content(title, api_key, model_name, prompts_list, provider_name, local_db_config)
                print(f"✅ Content generated for '{title}' using {model_name} via {provider_name}")
                break
            except Exception as e:
                print(f"⚠️ Content AI generation failed for '{title}' with {provider_name} ({model_name}): {e}. Trying next model for this provider.")
                ai_data = None

        if ai_data:
            break

    # ❌ SKIP fallback: don't insert into DB if AI completely failed
    if not ai_data:
        print(f"❌ All content AI models failed for '{title}'. Skipping this row.")
        log_ai_error(local_db_config, title, "N/A", "N/A", None, "All content AI models failed, skipping post.", str(prompts_list), "N/A")
        return None, None

    # ❌ SKIP known fallback content
    fallback_signature = f"This is fallback content generated when the AI did not return valid JSON for '{title}'"
    if fallback_signature in (ai_data.get('content') or ''):
        print(f"⚠️ Detected fallback content for '{title}', skipping insert.")
        log_ai_error(local_db_config, title, "Fallback", "N/A", None, "Fallback content detected and skipped.", str(prompts_list), ai_data.get('content', ''))
        return None, None

    # --- Image Generation (Optional) ---
    if image_ai_config.get('enabled', False):
        # The config now only contains the provider for this specific batch
        for provider_name, api_key_env_var, model_name in image_ai_config['providers']:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                print(f"❗ Skipping image generation with {provider_name} due to missing {api_key_env_var} for '{title}'.")
                continue

            current_image_models = [model_name] if not isinstance(model_name, list) else model_name
            for img_model_name in current_image_models:
                image_prompt = f"A high-quality, professional image for a blog post titled: '{title}'. Keywords: {ai_data.get('keywords', '')}. Summary: {ai_data.get('summary', '')}"
                try:
                    print(f"Attempting image generation for '{title}' with {provider_name} using model: {img_model_name}...")
                    generated_image_url = call_ai_api_for_image(image_prompt, api_key, img_model_name, provider_name, local_db_config)
                    print(f"🖼️ Image generated for '{title}' using {img_model_name} via {provider_name}: {generated_image_url}")
                    break
                except Exception as e:
                    print(f"⚠️ Image AI generation failed for '{title}' with {provider_name} ({img_model_name}): {e}. Trying next model.")
                    generated_image_url = ""
            if generated_image_url:
                break
    else:
        print("❗ Image generation skipped as 'image_ai_config['enabled']' is False.")

    # --- Prepare Tuple for DB ---
    final_category_id = int(selected_category_id) if selected_category_id else None
    final_user_id = int(selected_user_id) if selected_user_id else None
    final_domain_id = selected_domain_id
    now = datetime.now()

    post_data_tuple = (
        lang_id, title, ai_data.get('slug'), ai_data.get('title_hash'),
        ai_data.get('keywords'), ai_data.get('summary'), ai_data.get('content'),
        '',  # optional_url
        None, 0, 0, 0, 0, 0, 1, 0, 'article',
        '', '', generated_image_url, '', '', 1, None, '', 0,
        '', 0, 0, '', '', '', final_category_id, final_user_id,
        final_domain_id, now, now
    )

    return post_data_tuple, generated_image_url


def generate_rss_feed(posts_data, feed_category_name="Local Content", domain_label="Local Content", domain_slug="local-content"):
    # Utility: fallback slugify if not using a package
    def slugify(value):
        value = re.sub(r'[^\w\s-]', '', value.lower())
        return re.sub(r'[-\s]+', '-', value).strip('-_')

    now = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    rss_items = []

    # Clean base domain label (e.g., remove https:// or http://)
    clean_domain_label = domain_label.replace("https://", "").replace("http://", "").strip("/")
    base_url = f"https://{clean_domain_label}"
    full_feed_title = f"{clean_domain_label} - {feed_category_name} News Feed"

    for post in posts_data:
        # Title and slug
        title = html.escape(post.get('title') or 'Untitled')
        slug = post.get('slug') or slugify(title)

        # Description and content
        summary = html.escape(post.get('summary') or 'No summary provided.')
        content = post.get('content') or ''
        encoded_content = f"<![CDATA[{content}]]>"

        # Image tag if available
        image_url = post.get('image_url')
        image_tag = f'<img src="{html.escape(image_url)}" alt="{title}" style="max-width:100%;"/><br/>' if image_url else ''
        media_tag = f'<media:content url="{html.escape(image_url)}" medium="image" />' if image_url else ''

        # Date formatting
        created_at_raw = post.get('created_at')
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.utcnow()
        else:
            created_at = created_at_raw or datetime.utcnow()

        pub_date = created_at.strftime("%a, %d %b %Y %H:%M:%S +0000")

        # URL and metadata
        category_slug = post.get('category_slug') or 'uncategorized'
        link = f"{base_url}/{category_slug}/{slug}"
        guid = link

        # Final description
        full_description = f"{image_tag}{summary}"

        # Append RSS item
        rss_items.append(f"""
    <item>
      <title><![CDATA[{title}]]></title>
      <link><![CDATA[{link}]]></link>
      <description><![CDATA[{full_description}]]></description>
      <category><![CDATA[{feed_category_name}]]></category>
      <author><![CDATA[Unknown Author <noreply@example.com>]]></author>
      <guid isPermaLink="false"><![CDATA[{guid}]]></guid>
      <pubDate>{pub_date}</pubDate>
      <content:encoded>{encoded_content}</content:encoded>
      {media_tag.strip() if media_tag else ''}
    </item>
""")

    # Add fallback item if no posts exist
    if not rss_items:
        rss_items.append(f"""
    <item>
      <title>No Posts</title>
      <link>{base_url}</link>
      <description>No content available at the moment.</description>
      <guid>{base_url}</guid>
      <pubDate>{now}</pubDate>
    </item>
""")

    # Atom self link for RSS readers
    atom_slug = slugify(feed_category_name)
    atom_link = f"https://{domain_slug}/news-feeds/rss/{atom_slug}.xml"

    # Final RSS XML output
    rss_output = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:atom="http://www.w3.org/2005/Atom"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title><![CDATA[{full_feed_title}]]></title>
    <link><![CDATA[{base_url}]]></link>
    <description><![CDATA[RSS Feed for locally generated AI content for {feed_category_name} on {clean_domain_label}]]></description>
    <language>en</language>
    <pubDate>{now}</pubDate>
    <atom:link href="{atom_link}" rel="self" type="application/rss+xml"/>
    {''.join(rss_items)}
  </channel>
</rss>
"""
    return rss_output.strip()