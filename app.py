from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, Response
import os
import re
import requests
import csv
import uuid
import tempfile
import mysql.connector
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# Import necessary functions from uploader modules
from uploader.ai_content_generator import generate_content_from_prompt
from uploader.question_answerer import process_csv_and_answer
from uploader.upload_posts import upload_csv_with_file # Live DB upload
from uploader.local_content_generator import generate_and_save_local_posts, generate_rss_feed # New local content and RSS functions
from uploader.utils import slugify # Import slugify for public URLs
from flask import abort


load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = os.getenv('FLASK_SECRET_KEY')
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY environment variable not set! Please set it in your .env file.")

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    print(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")


# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Route to redirect to if user is not logged in

class User(UserMixin):
    """User class for Flask-Login."""
    def __init__(self, id, username, email, is_admin):
        self.id = id
        self.username = username
        self.email = email
        self.is_admin = is_admin

    def get_id(self):
        """Return the user ID as a string."""
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    """Loads a user from the database given their ID."""
    conn = None
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, is_admin FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
        if user_data:
            return User(id=user_data['id'], username=user_data['username'],
                        email=bool(user_data['email']), is_admin=bool(user_data['is_admin']))
    except Exception as e:
        print(f"Error loading user: {e}")
    finally:
        if conn:
            conn.close()
    return None
# --- End Flask-Login Setup ---


def get_local_db_connection():
    """Establishes a connection to the local database."""
    return mysql.connector.connect(
        host=os.getenv('LOCAL_DB_HOST'),
        user=os.getenv('LOCAL_DB_USER'),
        password=os.getenv('LOCAL_DB_PASSWORD'),
        database=os.getenv('LOCAL_DB_NAME')
    )

def get_dynamic_db_configs():
    """Fetches dynamic database configurations from the local db_targets table."""
    conn = None
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # ✅ Include host directly in the top-level dict
        cursor.execute("SELECT id, db_key, host, user, password, database_name, label FROM db_targets")
        rows = cursor.fetchall()

        return {
            row['db_key']: {
                'id': row['id'],
                'host': row['host'],  # <-- make host available for templates
                'label': row.get('label', row['host']),
                'connection': {
                    'host': row['host'],
                    'user': row['user'],
                    'password': row['password'],
                    'database': row['database_name']
                }
            } for row in rows
        }

    except Exception as e:
        print(f"Error loading DB configs: {e}")
        return {}
    finally:
        if conn:
            conn.close()


def get_db_connection(db_key):
    """Establishes a connection to a dynamically configured database."""
    configs = get_dynamic_db_configs()
    config_entry = configs.get(db_key)

    # Ensure config_entry is not None and has the 'connection' key
    if not config_entry or 'connection' not in config_entry:
        raise Exception(f"Database configuration for '{db_key}' not found or is incomplete.")

    return mysql.connector.connect(**config_entry['connection'])

def get_domain_configs():
    """Fetches domain host data from the local db_targets table."""
    connection = None
    domain_hosts_data = {}
    try:
        connection = mysql.connector.connect(
            host=os.getenv('LOCAL_DB_HOST'),
            user=os.getenv('LOCAL_DB_USER'),
            password=os.getenv('LOCAL_DB_PASSWORD'),
            database=os.getenv('LOCAL_DB_NAME')
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT db_key, host, label FROM db_targets ORDER BY label")
        for row in cursor.fetchall():
            domain_hosts_data[row['db_key']] = {
                'host': row['host'],
                'label': row['label'] if row['label'] is not None else row['host'] # Ensure label is not None
            }
    except Exception as e:
        print(f"Error fetching domain configs: {str(e)}")
    finally:
        if connection:
            connection.close()
    return domain_hosts_data

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = None
        try:
            conn = get_local_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username, password_hash, email, is_admin FROM users WHERE username = %s", (username,))
            user_data = cursor.fetchone()

            if user_data and check_password_hash(user_data['password_hash'], password):
                user = User(id=user_data['id'], username=user_data['username'],
                            email=bool(user_data['email']), is_admin=bool(user_data['is_admin']))
                login_user(user)
                flash('Logged in successfully!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('admin_dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
        except Exception as e:
            flash(f'An error occurred during login: {e}', 'danger')
            print(f"Login error: {e}")
        finally:
            if conn:
                conn.close()
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html', username=username, email=email)

        conn = None
        try:
            conn = get_local_db_connection()
            cursor = conn.cursor(dictionary=True)

            # Check if username or email already exists
            cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()
            if existing_user:
                flash('Username or email already exists. Please choose a different one.', 'danger')
                return render_template('register.html', username=username, email=email)

            hashed_password = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (username, password_hash, email, is_admin) VALUES (%s, %s, %s, %s)",
                (username, hashed_password, email, 0) # Default new users to not admin
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'An error occurred during registration: {e}', 'danger')
            print(f"Registration error: {e}")
            conn.rollback() # Rollback in case of error
        finally:
            if conn:
                conn.close()
    return render_template('register.html')


@app.route('/logout')
@login_required # Ensure only logged in users can log out
def logout():
    """Handles user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/')
def home_redirect():
    """Redirects base URL to the admin dashboard."""
    return redirect(url_for('admin_dashboard'))

@app.route('/admin', methods=['GET'])
@login_required # Protect this route
def admin_dashboard():
    """Renders the main admin dashboard page."""
    return render_template('admin_dashboard.html')

@app.route('/admin/upload_posts', methods=['GET', 'POST'])
@login_required # Protect this route
def upload_posts():
    """Handles the CSV upload and AI post generation."""
    db_hosts_data = get_dynamic_db_configs()
    domain_hosts_data = get_domain_configs()

    if request.method == 'POST':
        db_host_key = request.form.get('domain_id')
        csv_file = request.files.get('csv_file')
        raw_prompts_string = request.form.get('prompts')
        category_id = request.form.get('category_id')
        user_id = request.form.get('user_id') # CORRECTED: Changed from 'user' to 'user_id'

        prompts_list = [prompt.strip() for prompt in re.split(r'\n\s*\n|\n', raw_prompts_string) if prompt.strip()] if raw_prompts_string else []

        if not csv_file or csv_file.filename == '':
            flash('Please upload a CSV file.', 'danger')
            return redirect(url_for('upload_posts'))

        # Retrieve the specific config entry for the selected host
        db_config_entry = db_hosts_data.get(db_host_key)

        # Validate that the config entry exists and contains the 'connection' key
        if not db_config_entry or 'connection' not in db_config_entry:
            flash('Invalid database selection or configuration missing connection details.', 'danger')
            return redirect(url_for('upload_posts'))

        # Extract only the connection parameters to pass
        db_connection_params = db_config_entry['connection']

        if not category_id:
            flash('Please select a Category.', 'danger')
            return redirect(url_for('upload_posts'))
        if not user_id:
            flash('Please select a User.', 'danger')
            return redirect(url_for('upload_posts'))

        content_ai_providers_config = {
            'providers': [
                ('together_ai', 'TOGETHER_API_KEY', [
                    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'zero-one-ai/Yi-34B-Chat',
                    'togethercomputer/Llama-2-70B-32K-Instruct',
                    'databricks/dbrx-instruct',
                    'Qwen/Qwen1.5-72B-Chat',
                    'DiscoResearch/DiscoLM-70b-v2'
                ]),
                ('aiml_api', 'AIML_API_KEY', [
                    'gpt-3.5-turbo',
                    'gpt-4',
                    'text-davinci-003',
                    'gpt-3.5-turbo-instruct',
                    'gpt-4o'
                ])
            ]
        }

        image_ai_providers_config = {
            'providers': [
                ('together_ai', 'TOGETHER_API_KEY', 'black-forest-labs/FLUX.1-kontext-pro'),
            ],
            'enabled': True
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            csv_file.save(tmp.name)
            csv_path = tmp.name

        try:
            # Construct local_db_config from environment variables
            local_db_config = {
                'host': os.getenv('LOCAL_DB_HOST'),
                'user': os.getenv('LOCAL_DB_USER'),
                'password': os.getenv('LOCAL_DB_PASSWORD'),
                'database': os.getenv('LOCAL_DB_NAME')
            }

            # Pass the db_host_key (selected_domain_id) to upload_csv_with_file for logging
            upload_csv_with_file(
                csv_path=csv_path,
                db_config=db_connection_params,
                content_ai_config=content_ai_providers_config,
                image_ai_config=image_ai_providers_config,
                prompts_list=prompts_list,
                local_db_config=local_db_config, # Pass the local_db_config here
                selected_category_id=int(category_id),
                selected_user_id=int(user_id),
                selected_domain_id=db_host_key # Pass the db_host_key for logging
            )
            flash('CSV uploaded and processed successfully!', 'success')
            return redirect(url_for('show_results', db_host=db_host_key))
        except Exception as e:
            flash(f'Error during upload: {str(e)}', 'danger')
            print(f"Error during upload: {str(e)}")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

        return redirect(url_for('upload_posts')) # CORRECTED: Changed from 'result' to 'upload_posts'

    return render_template(
        'upload_posts.html',
        db_hosts_data=db_hosts_data,
        domain_hosts_data=domain_hosts_data
    )

@app.route('/admin/generate_local_posts', methods=['GET', 'POST'])
@login_required
def generate_local_posts_route():
    """
    Handles the local CSV upload, AI post generation for local DB, and RSS feed generation.
    """
    domain_configs = get_domain_configs()  # Use for displaying domain labels

    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        raw_prompts_string = request.form.get('prompts')
        category_id = request.form.get('category_id')
        user_id = request.form.get('user_id')
        selected_db_key = request.form.get('domain_id_for_save')
        selected_domain_label = request.form.get('domain_label_for_rss')

        prompts_list = [prompt.strip() for prompt in re.split(r'\n\s*\n|\n', raw_prompts_string) if prompt.strip()] if raw_prompts_string else []

        if not csv_file or csv_file.filename == '':
            flash('Please upload a CSV file.', 'danger')
            return redirect(url_for('generate_local_posts_route'))
        if not category_id or not user_id or not selected_db_key:
            flash('Please select all required fields: Category, User, and Domain.', 'danger')
            return redirect(url_for('generate_local_posts_route'))

        # Resolve numeric domain_id using db_key
        domain_numeric_id = None
        try:
            conn = get_local_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, label FROM db_targets WHERE db_key = %s", (selected_db_key,))
            domain_info = cursor.fetchone()
            if domain_info:
                domain_numeric_id = domain_info['id']
                if not selected_domain_label:
                    selected_domain_label = domain_info['label']
            else:
                flash("Invalid domain selected.", "danger")
                return redirect(url_for('generate_local_posts_route'))
        except Exception as e:
            flash(f"Database error while resolving domain: {e}", "danger")
            return redirect(url_for('generate_local_posts_route'))
        finally:
            if conn:
                conn.close()

        # Configs for AI content/image generation
        # Updated content_ai_providers_config with AIML API as the first provider
        content_ai_providers_config = {
            'providers': [
                ('aiml_api', 'AIML_API_KEY', [
                    'gpt-3.5-turbo',
                    'gpt-4',
                    'text-davinci-003',
                    'gpt-3.5-turbo-instruct',
                    'gpt-4o'
                ]),
                ('together_ai', 'TOGETHER_API_KEY', [
                    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
                    'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'zero-one-ai/Yi-34B-Chat',
                    'togethercomputer/Llama-2-70B-32K-Instruct',
                    'databricks/dbrx-instruct',
                    'Qwen/Qwen1.5-72B-Chat',
                    'DiscoResearch/DiscoLM-70b-v2'
                ])
            ]
        }

        image_ai_providers_config = {
            'providers': [
                ('aiml_api', 'AIML_API_KEY', 'flux/schnell'),
                ('together_ai', 'TOGETHER_API_KEY', 'black-forest-labs/FLUX.1-dev')
            ],
            'enabled': True
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            csv_file.save(tmp.name)
            csv_path = tmp.name

        try:
            local_db_config = {
                'host': os.getenv('LOCAL_DB_HOST'),
                'user': os.getenv('LOCAL_DB_USER'),
                'password': os.getenv('LOCAL_DB_PASSWORD'),
                'database': os.getenv('LOCAL_DB_NAME')
            }

            generated_posts_for_rss = generate_and_save_local_posts(
                csv_path=csv_path,
                content_ai_config=content_ai_providers_config,
                image_ai_config=image_ai_providers_config,
                prompts_list=prompts_list,
                local_db_config=local_db_config,
                selected_category_id=int(category_id),
                selected_user_id=int(user_id),
                selected_domain_label=selected_domain_label,
                selected_domain_id=domain_numeric_id  # <- Use actual numeric ID for FK
            )

            if generated_posts_for_rss:
                rss_link = url_for('download_local_rss', category_id=category_id, domain_id=selected_db_key)
                flash(f'Local posts generated and RSS feed created successfully! '
                      f'<a href="{rss_link}" class="font-bold underline">Download RSS Feed</a>.', 'success')

                # Redirect to category feed view
                return redirect(url_for('admin_rss_category_index', selected_domain_id=selected_db_key))

            flash('No local posts were generated.', 'warning')
            return redirect(url_for('generate_local_posts_route'))

        except Exception as e:
            flash(f'Error during local content generation: {str(e)}', 'danger')
            print(f"[ERROR] generate_local_posts_route: {e}")
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

        return redirect(url_for('generate_local_posts_route'))

    # Handle GET request to render form
    conn = None
    categories, users = [], []
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name FROM categories ORDER BY name")
        categories = cursor.fetchall()
        cursor.execute("SELECT id, username FROM users ORDER BY username")
        users = cursor.fetchall()
    except Exception as e:
        flash(f"Error loading form data: {e}", "danger")
    finally:
        if conn:
            conn.close()

    return render_template(
        'generate_local_posts.html',
        db_hosts_data=domain_configs,
        domain_hosts_data=domain_configs,
        categories=categories,
        users=users
    )

# REMOVED: @app.route('/admin/filter_local_posts', methods=['GET', 'POST'])
# REMOVED: def filter_local_posts():
# REMOVED:     """Renders a page allowing users to select a domain to filter local posts."""
# REMOVED:     # ... (logic removed as it's now in local_posts_results)

@app.route('/rss-feed/<string:host>/<int:category_id>', methods=['GET'])
def rss_feed_by_domain_and_category(host, category_id):
    """
    Generates an RSS feed using domain host and category ID.
    """
    conn = None
    posts = []
    domain_label = "Unknown Domain"
    category_name = "Unknown Category"
    category_slug = "uncategorized"

    try:
        # 🔌 Connect to local DB
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 🔍 Lookup domain by host
        cursor.execute("SELECT id, label FROM db_targets WHERE host = %s", (host,))
        domain_result = cursor.fetchone()
        if not domain_result:
            return Response(f"<error>Invalid host '{host}'</error>", mimetype='application/rss+xml', status=404)

        numeric_domain_id = domain_result['id']
        domain_label = domain_result.get('label') or host

        # 🔍 Lookup category by ID
        cursor.execute("SELECT name, slug FROM categories WHERE id = %s", (category_id,))
        category_result = cursor.fetchone()
        if category_result:
            category_name = category_result.get('name') or category_name
            category_slug = category_result.get('slug') or category_slug

        # 📦 Fetch posts
        cursor.execute("""
            SELECT title, summary, content, image_url, created_at, slug, category_id, domain_id
            FROM local_generated_posts
            WHERE domain_id = %s AND category_id = %s
            ORDER BY created_at DESC
        """, (numeric_domain_id, category_id))
        posts = cursor.fetchall()

        if not posts:
            print(f"⚠️ No posts found for host '{host}' and category '{category_id}'")
            return Response("", status=204)

        # 🏷️ Add category slug to each post
        for post in posts:
            post['category_slug'] = category_slug

        # 🧠 Generate RSS feed XML
        from uploader.local_content_generator import generate_rss_feed
        rss_xml = generate_rss_feed(
            posts,
            feed_category_name=category_name,
            domain_label=domain_label,
            domain_slug=host  # Used to build <link> and atom:link
        )

        # ✅ Return proper RSS response
        return Response(rss_xml, mimetype='application/xml')

    except Exception as e:
        print(f"❌ Error generating RSS feed for host '{host}' and category '{category_id}': {e}")
        return Response("<error>Unable to generate RSS feed</error>", mimetype='application/xml', status=500)

    finally:
        if conn:
            conn.close()


# --- Updated Local Posts Results Route ---
@app.route('/admin/local_posts_results', methods=['GET'])
@login_required
def local_posts_results():
    """
    Displays locally generated posts, optionally filtered by domain and category.
    Also shows per-category feed links and RSS download.
    """
    selected_domain_id = request.args.get('selected_domain_id')
    selected_category_id = request.args.get('selected_category_id', type=int)

    conn = None
    posts = []
    category_name = None
    domain_label = None
    categories_with_posts = []
    domain_hosts_data = get_domain_configs()

    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get category name if category is selected
        if selected_category_id:
            cursor.execute("SELECT name FROM categories WHERE id = %s", (selected_category_id,))
            cat_info = cursor.fetchone()
            if cat_info and cat_info['name']:
                category_name = cat_info['name']
            else:
                print(f"[WARN] Category ID {selected_category_id} not found.")

        # Get domain label if selected
        if selected_domain_id:
            domain_config = domain_hosts_data.get(selected_domain_id)
            domain_label = domain_config.get('label') if domain_config else selected_domain_id
            if not domain_config:
                print(f"[WARN] Domain ID {selected_domain_id} not found. Using ID as fallback.")

        # Fetch recent posts with optional filtering
        query = """
            SELECT l.id, l.title, l.summary, l.content, l.image_url, l.created_at, l.slug,
                   c.name AS category_name, c.slug AS category_slug,
                   d.label AS domain_label, d.db_key AS domain_id
            FROM local_generated_posts AS l
            JOIN categories AS c ON l.category_id = c.id
            JOIN db_targets AS d ON l.domain_id = d.db_key
            WHERE 1=1
        """
        params = []

        if selected_domain_id:
            query += " AND l.domain_id = %s"
            params.append(selected_domain_id)
        if selected_category_id:
            query += " AND l.category_id = %s"
            params.append(selected_category_id)

        query += " ORDER BY l.created_at DESC LIMIT 20"
        cursor.execute(query, params)
        posts = cursor.fetchall()

        # Fetch categories with post count for this domain
        if selected_domain_id:
            cursor.execute("""
                SELECT c.id, c.name, c.slug, COUNT(l.id) AS post_count
                FROM categories AS c
                JOIN local_generated_posts AS l ON c.id = l.category_id
                WHERE l.domain_id = %s
                GROUP BY c.id, c.name, c.slug
                ORDER BY c.name
            """, (selected_domain_id,))
            categories_with_posts = cursor.fetchall()

    except Exception as e:
        flash(f"Error loading local posts results: {str(e)}", 'danger')
        print(f"[ERROR] local_posts_results: {e}")
    finally:
        if conn:
            conn.close()

    return render_template(
        'local_results.html',
        posts=posts,
        selected_domain_id=selected_domain_id,
        selected_domain_label=domain_label,
        selected_category_id=selected_category_id,
        category_name=category_name,
        domain_hosts_data=domain_hosts_data,
        categories_with_posts=categories_with_posts
    )

# NEW PUBLIC ROUTES
@app.route('/news-feeds', methods=['GET'])
def public_rss_categories():
    """Displays a public list of categories for RSS feeds, with an optional domain filter."""
    selected_domain_id = request.args.get('selected_domain_id') # Get domain_id from query params

    conn = None
    categories = []
    domain_hosts_data = get_domain_configs() # Get all domain configs for the dropdown
    selected_domain_label = None

    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get domain label for display if selected_domain_id is provided
        if selected_domain_id:
            domain_config = domain_hosts_data.get(selected_domain_id)
            if domain_config and domain_config.get('label') is not None:
                selected_domain_label = domain_config.get('label')
            else:
                selected_domain_label = "All Domains" # Fallback if ID is invalid or label missing
                print(f"Warning: Domain ID {selected_domain_id} not found or label is NULL. Displaying 'All Domains'.")
                # Reset selected_domain_id if it's invalid to show all relevant categories
                selected_domain_id = None 


        # Base query to select distinct categories that have posts
        query = """
            SELECT DISTINCT c.id, c.name, c.slug 
            FROM categories AS c
            JOIN local_generated_posts AS l ON c.id = l.category_id
        """
        params = []

        if selected_domain_id:
            # If a domain is selected, filter categories to only show those with posts in that domain
            query += " WHERE l.domain_id = %s"
            params.append(selected_domain_id)
        
        query += " ORDER BY c.name" # Order for consistent display

        cursor.execute(query, params)
        categories = cursor.fetchall()

    except Exception as e:
        print(f"Error loading public categories: {e}")
        flash(f"Error loading categories: {str(e)}", "danger")
    finally:
        if conn:
            conn.close()
    
    # Pass domain_hosts_data and selected_domain_id/label to the template
    return render_template('public_rss_categories.html', 
                           categories=categories,
                           domain_hosts_data=domain_hosts_data,
                           selected_domain_id=selected_domain_id,
                           selected_domain_label=selected_domain_label)


@app.route('/news-feeds/<string:category_slug>', methods=['GET'])
def public_category_posts(category_slug):
    """Displays public posts for a specific category."""
    conn = None
    posts = []
    category_name = "Unknown Category"
    category_id = None # To fetch posts by ID

    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # First, find the category_id and name based on the slug
        cursor.execute("SELECT id, name FROM categories WHERE slug = %s", (category_slug,))
        category_info = cursor.fetchone()

        if category_info and category_info['name'] is not None: # Ensure name is not None
            category_id = category_info['id']
            category_name = category_info['name']
            # Fetch posts for this category, including their slugs
            cursor.execute("SELECT title, summary, content, image_url, created_at, slug FROM local_generated_posts WHERE category_id = %s ORDER BY created_at DESC", (category_id,))
            posts = cursor.fetchall()
        else:
            flash(f"Category '{category_slug.replace('-', ' ').title()}' not found.", "warning")
            return redirect(url_for('public_rss_categories'))

    except Exception as e:
        print(f"Error loading public category posts for '{category_slug}': {e}")
        flash(f"Error loading posts: {str(e)}", "danger")
    finally:
        if conn:
            conn.close()
    return render_template('public_category_posts.html', posts=posts, category_name=category_name, category_slug=category_slug)

@app.route('/news-feeds/<string:category_slug>/<string:post_slug>', methods=['GET'])
def public_single_post(category_slug, post_slug):
    """Displays a single locally generated post on the public side."""
    conn = None
    post = None
    category_name = "Unknown Category"
    
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get category_id from category_slug
        cursor.execute("SELECT id, name FROM categories WHERE slug = %s", (category_slug,))
        category_info = cursor.fetchone()

        if category_info and category_info['name'] is not None: # Ensure name is not None
            category_id = category_info['id']
            category_name = category_info['name']
            
            # Fetch the specific post by category_id and post_slug
            cursor.execute("""
                SELECT title, summary, content, image_url, created_at
                FROM local_generated_posts
                WHERE category_id = %s AND slug = %s
                LIMIT 1
            """, (category_id, post_slug))
            post = cursor.fetchone()
        else:
            flash(f"Category '{category_slug.replace('-', ' ').title()}' not found.", "warning")
            return redirect(url_for('public_rss_categories'))

        if not post:
            flash(f"Post '{post_slug.replace('-', ' ').title()}' not found in category '{category_name}'.", "warning")
            return redirect(url_for('public_category_posts', category_slug=category_slug))

    except Exception as e:
        print(f"Error loading public single post '{post_slug}' in category '{category_slug}': {e}")
        flash(f"Error loading post: {str(e)}", "danger")
    finally:
        if conn:
            conn.close()
            
    return render_template('public_single_post.html', post=post, category_name=category_name, category_slug=category_slug)

@app.route('/news-feeds/rss/<string:category_slug>.xml', methods=['GET'])
def public_download_rss_category(category_slug):
    """Allows public viewing of category-specific RSS feed files, with optional domain filter."""
    selected_domain_id = request.args.get('selected_domain_id') # Get domain_id from query params

    posts_to_include = []
    feed_category_name = "Local Content" # Default label for the RSS feed title
    category_id = None
    domain_label_for_rss = "news-feeds.example.com" # Default domain for RSS links
    domain_slug_for_rss = "news-feeds-example-com" # Default slug for RSS atom:link

    conn = None
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Find category_id and name based on slug
        cursor.execute("SELECT id, name FROM categories WHERE slug = %s", (category_slug,))
        category_info = cursor.fetchone()
        
        if category_info and category_info['name'] is not None:
            category_id = category_info['id']
            feed_category_name = category_info['name']
        else:
            print(f"Category slug '{category_slug}' not found for RSS feed generation. Returning empty feed.")
            return Response(generate_rss_feed([], "Category Not Found", "news-feeds.example.com", category_slug), mimetype='application/rss+xml')

        # Get domain label for RSS title and links if selected_domain_id is provided
        if selected_domain_id:
            domain_configs = get_domain_configs()
            domain_config = domain_configs.get(selected_domain_id)
            if domain_config and domain_config.get('label') is not None:
                domain_label_for_rss = domain_config.get('label')
                domain_slug_for_rss = slugify(domain_label_for_rss)
            else:
                print(f"Warning: Domain ID {selected_domain_id} not found or label is NULL. Using default for RSS feed.")


        # Build query for posts
        query = """
            SELECT l.title, l.summary, l.content, l.image_url, l.slug, l.created_at, c.slug AS category_slug, d.label AS domain_label
            FROM local_generated_posts AS l
            JOIN categories AS c ON l.category_id = c.id
            JOIN db_targets AS d ON l.domain_id = d.db_key
            WHERE l.category_id = %s
        """
        params = [category_id]

        if selected_domain_id:
            query += " AND l.domain_id = %s"
            params.append(selected_domain_id)
        
        query += " ORDER BY l.created_at DESC" # Order for RSS feed

        cursor.execute(query, params)
        posts_to_include = cursor.fetchall()

        if not posts_to_include:
            print(f"No posts found for category '{feed_category_name}' (ID: {category_id}) and domain '{selected_domain_id or 'All'}' for RSS feed.")
            # Return a valid but empty RSS feed if no posts found
            return Response(generate_rss_feed([], feed_category_name, domain_label_for_rss, category_slug), mimetype='application/rss+xml')

        rss_feed_content = generate_rss_feed(posts_to_include, feed_category_name, domain_label_for_rss, domain_slug_for_rss)
        
        return Response(rss_feed_content, mimetype='application/rss+xml')

    except Exception as e:
        print(f"Error generating public category RSS feed: {str(e)}")
        error_rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Error Generating RSS Feed</title>
        <link>http://localhost:5000/news-feeds</link>
        <description>An error occurred while generating the RSS feed. Details: {str(e)}</description>
    </channel>
</rss>"""
        return Response(error_rss, mimetype='application/rss+xml')
    finally:
        if conn:
            conn.close()


# ADMIN RSS Download Route (now also for viewing directly)
@app.route('/download_local_rss', methods=['GET'])
@login_required
def download_local_rss():
    """Allows viewing of locally generated RSS feed files, optionally filtered by category and domain (ADMIN ONLY)."""
    category_id = request.args.get('category_id', type=int)
    domain_id = request.args.get('domain_id') # Get the domain_id from request arguments
    
    posts_to_include = []
    feed_domain_label = "Local Content"
    feed_category_name = "All Posts" # Default for RSS title if no category is selected
    category_slug_for_rss = "all-posts" # Default for slug in RSS atom:link
    domain_slug_for_rss = "local-content" # Default slug for RSS atom:link

    conn = None
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get actual category name and slug if ID is provided
        if category_id:
            cursor.execute("SELECT name, slug FROM categories WHERE id = %s", (category_id,))
            cat_info = cursor.fetchone()
            if cat_info and cat_info['name'] is not None:
                feed_category_name = cat_info['name']
                category_slug_for_rss = cat_info['slug']
            else:
                print(f"Warning: Category ID {category_id} not found or name is NULL when generating RSS for admin download.")

        # Get domain label for RSS title and links
        if domain_id:
            db_configs = get_dynamic_db_configs()
            domain_config = db_configs.get(domain_id)
            if domain_config and domain_config.get('label') is not None:
                feed_domain_label = domain_config.get('label')
                domain_slug_for_rss = slugify(feed_domain_label) 
            else:
                feed_domain_label = domain_id if domain_id else "Unknown Domain"
                domain_slug_for_rss = slugify(feed_domain_label)
                print(f"Warning: Domain ID {domain_id} not found or label is NULL. Using '{feed_domain_label}' as label.")

        # Build query for posts
        query = """
            SELECT l.title, l.summary, l.content, l.image_url, l.slug, l.created_at, c.slug AS category_slug, d.label AS domain_label
            FROM local_generated_posts AS l
            JOIN categories AS c ON l.category_id = c.id
            JOIN db_targets AS d ON l.domain_id = d.db_key
            WHERE 1=1
        """
        params = []
        if domain_id:
            query += " AND l.domain_id = %s"
            params.append(domain_id)
        if category_id:
            query += " AND l.category_id = %s"
            params.append(category_id)
        
        query += " ORDER BY l.created_at DESC LIMIT 100" # Limit for general feed

        cursor.execute(query, params)
        posts_to_include = cursor.fetchall()

        if not posts_to_include:
            flash(f"No posts found for RSS feed (Category: {feed_category_name}, Domain: {feed_domain_label}).", "warning")
            rss_feed_content = generate_rss_feed([], feed_category_name, feed_domain_label, domain_slug_for_rss)
            return Response(rss_feed_content, mimetype='application/rss+xml')

        rss_feed_content = generate_rss_feed(posts_to_include, feed_category_name, feed_domain_label, domain_slug_for_rss)
        
        return Response(rss_feed_content, mimetype='application/rss+xml')

    except Exception as e:
        flash(f"Error downloading RSS feed: {str(e)}", "danger")
        print(f"Error serving RSS file: {e}")
        error_rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Error Generating RSS Feed</title>
        <link>http://localhost:5000/admin/local_posts_results</link>
        <description>An error occurred while generating the RSS feed. Details: {str(e)}</description>
    </channel>
</rss>"""
        return Response(error_rss, mimetype='application/rss+xml')
    finally:
        if conn:
            conn.close()


@app.route('/admin/upload-questions', methods=['GET', 'POST'])
@login_required # Protect this route
def upload_questions():
    """Handles the upload and processing of question CSVs."""
    db_hosts_data = get_dynamic_db_configs()

    if request.method == 'POST':
        db_host_key = request.form.get('domain_id')

        # Validate db_host_key and its configuration before proceeding
        db_config_entry = db_hosts_data.get(db_host_key)
        if not db_config_entry or 'connection' not in db_config_entry:
            flash('Invalid database selected or configuration missing connection details.', 'danger')
            return redirect(url_for('upload_questions'))

        if 'csv_file' not in request.files:
            flash('No file part.', 'danger')
            return redirect(url_for('upload_questions'))

        file = request.files['csv_file']
        if file.filename == '':
            flash('No selected file.', 'danger')
            return redirect(url_for('upload_questions'))

        try:
            result = process_csv_and_answer(file, db_key=db_host_key)
            # Use get_db_connection which now has robust validation
            conn = get_db_connection(db_host_key)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username FROM users")
            users_map = {str(row['id']): row['username'] for row in cursor.fetchall()}
            conn.close()

            return render_template('question_result.html', result=result, users_map=users_map)
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('upload_questions'))

    return render_template('upload_questions.html', db_hosts_data=db_hosts_data)

@app.route('/admin/results')
@login_required # Protect this route
def show_results():
    """Displays the results of post uploads."""
    db_host_key = request.args.get('db_host')
    db_hosts_data = get_dynamic_db_configs()
    db_config_entry = db_hosts_data.get(db_host_key) # Get the entire config entry

    conn = None
    posts = []
    try:
        # Validate the config entry and extract connection parameters
        if not db_config_entry or not isinstance(db_config_entry, dict) or 'connection' not in db_config_entry:
            flash("Error loading results: Invalid or missing database configuration for the selected host.", 'danger')
            # Return an empty list of posts if configuration is invalid
            return render_template('result.html', posts=[], generated_image_urls=[])

        connection_params = db_config_entry['connection'] # Extract the 'connection' sub-dictionary

        conn = mysql.connector.connect(**connection_params) # Use the extracted connection parameters
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT title, summary, content, image_url, created_at FROM posts ORDER BY created_at DESC LIMIT 20")
        posts = cursor.fetchall()
    except Exception as e:
        flash(f"Error loading results: {str(e)}", 'danger')
    finally:
        if conn:
            conn.close()
    return render_template('result.html', posts=posts, generated_image_urls=[])

@app.route('/admin/upload-db-configs', methods=['GET', 'POST'])
@login_required # Protect this route
def upload_db_configs():
    """Handles the upload of database configuration CSVs."""
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or file.filename == '':
            flash("Please upload a CSV file.", "danger")
            return redirect(url_for('upload_db_configs'))

        try:
            conn = get_local_db_connection()
            cursor = conn.cursor()

            reader = csv.DictReader(file.stream.read().decode("utf-8").splitlines())
            for row in reader:
                cursor.execute("""
                    INSERT INTO db_targets (db_key, host, user, password, database_name, label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    host = VALUES(host),
                    user = VALUES(user),
                    password = VALUES(password),
                    database_name = VALUES(database_name),
                    label = VALUES(label)
                """, (
                    row['db_key'],
                    row['host'],
                    row['user'],
                    row['password'],
                    row['database_name'],
                    row.get('label')
                ))

            conn.commit()
            flash("Database configurations uploaded successfully!", "success")
            return redirect(url_for('admin_dashboard')) # Redirect to admin dashboard
        except Exception as e:
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for('upload_db_configs'))
        finally:
            if conn:
                conn.close()

    return render_template('upload_db.html')

@app.route('/admin/upload-image', methods=['GET', 'POST'])
@login_required
def upload_image():
    """Generates an AI image from a logo and prompt using Together AI with logging."""
    uploaded_image_url = None
    generated_image_url = None

    if request.method == 'POST':
        image = request.files.get('image')
        prompt = request.form.get('prompt', '').strip()

        print("\n[INFO] === Received Upload Request ===")
        print(f"[INFO] Prompt: {prompt}")
        print(f"[INFO] Image present: {bool(image and image.filename)}")

        if not prompt:
            flash(('danger', 'Prompt is required.'))
            return redirect(request.url)

        uploaded_image_path = None
        if image and image.filename:
            filename = secure_filename(image.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            image.save(uploaded_image_path)
            uploaded_image_url = url_for('static', filename=f'uploads/{unique_filename}', _external=True)
            print(f"[INFO] Saved uploaded image to: {uploaded_image_path}")
            print(f"[INFO] Accessible image URL: {uploaded_image_url}")

        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "black-forest-labs/FLUX.1-kontext-dev",
                "prompt": prompt,
                "image_url": uploaded_image_url
            }

            print(f"[DEBUG] Payload to Together AI:\n{payload}")
            print("[DEBUG] Sending POST request to Together AI...")

            response = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers=headers,
                json=payload
            )

            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response text: {response.text}")

            if response.status_code != 200:
                flash(('danger', f"API Error: {response.status_code} - {response.text}"))
                return redirect(request.url)

            result = response.json()
            image_url = result.get('image_url') or result.get('output', [{}])[0].get('image_url')

            print(f"[INFO] Returned image URL: {image_url}")

            if not image_url:
                flash(('warning', 'No image returned from AI.'))
                return redirect(request.url)

            download_resp = requests.get(image_url)
            if download_resp.status_code == 200:
                gen_filename = f"{uuid.uuid4().hex}_generated.png"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], gen_filename)
                with open(save_path, 'wb') as f:
                    f.write(download_resp.content)
                generated_image_url = url_for('static', filename=f'uploads/{gen_filename}')
                print(f"[INFO] Saved generated image to: {save_path}")
            else:
                flash(('warning', 'Failed to download generated image.'))

        except Exception as e:
            print(f"[ERROR] Exception during generation: {str(e)}")
            flash(('danger', f"An error occurred: {str(e)}"))
            return redirect(request.url)

    return render_template(
        'upload_image.html',
        uploaded_image_url=uploaded_image_url,
        generated_image_url=generated_image_url
    )

@app.route('/get_categories_by_db_key/<string:db_key>', methods=['GET'])
@login_required # Protect this route
def get_categories_by_db_key(db_key):
    """Fetches categories for a given database key."""
    conn = None
    try:
        # Get the full config entry and validate
        configs = get_dynamic_db_configs()
        config_entry = configs.get(db_key)
        if not config_entry or 'connection' not in config_entry:
            return jsonify({'error': 'Invalid database key or configuration missing connection details'}), 400

       
        conn = mysql.connector.connect(**config_entry['connection'])
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT id, name FROM categories ORDER BY name")
        rows = cursor.fetchall()

        categories = [{'id': row['id'], 'name': row['name']} for row in rows]

        return jsonify({'categories': categories})

    except Exception as e:
        print(f"Error fetching categories for DB key '{db_key}': {e}")
        return jsonify({'error': 'Could not fetch categories', 'details': str(e)}), 500

    finally:
        if conn:
            conn.close()

@app.route('/<string:domain_label>/<string:category_slug>', methods=['GET'])
def public_domain_category_view(domain_label, category_slug):
    """
    View posts for a selected domain and category.
    URL pattern: http://127.0.0.1:5000/<host>/<category_slug>
    Example: http://127.0.0.1:5000/geetanshmehra.com/technology
    """
    conn = None
    posts = []
    category = None
    domain_id = None

    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Step 1: Find domain_id by domain label (like 'geetanshmehra.com')
        cursor.execute("SELECT id, label FROM db_targets WHERE label = %s", (domain_label,))
        domain_data = cursor.fetchone()
        if not domain_data:
            return abort(404, f"Domain '{domain_label}' not found")
        domain_id = domain_data['id']

        # Step 2: Get category by slug
        cursor.execute("SELECT id, name, slug FROM categories WHERE slug = %s", (category_slug,))
        category = cursor.fetchone()
        if not category:
            return abort(404, f"Category '{category_slug}' not found")

        # Step 3: Fetch posts from local_generated_posts filtered by domain and category
        cursor.execute("""
            SELECT title, summary, content, slug, image_url, created_at
            FROM local_generated_posts
            WHERE domain_id = %s AND category_id = %s AND status = 1
            ORDER BY created_at DESC
            LIMIT 20
        """, (domain_id, category['id']))
        posts = cursor.fetchall()

    except Exception as e:
        print(f"[ERROR] public_domain_category_view: {e}")
        return abort(500)
    finally:
        if conn:
            conn.close()

    return render_template("public_domain_category_feed.html",
                           posts=posts,
                           category=category,
                           domain_label=domain_label)

@app.route('/get_users_by_db_key/<string:db_key>', methods=['GET'])
@login_required
def get_users_by_db_key(db_key):
    """Fetches users for a given database key."""
    conn = None
    try:
        # Get connection config
        configs = get_dynamic_db_configs()
        config_entry = configs.get(db_key)
        if not config_entry or 'connection' not in config_entry:
            return jsonify({'error': 'Invalid database key or configuration missing'}), 400

        # Connect and fetch users
        conn = mysql.connector.connect(**config_entry['connection'])
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username FROM users ORDER BY username")
        users = cursor.fetchall()

        return jsonify({'users': users})

    except Exception as e:
        print(f"[ERROR] get_users_by_db_key({db_key}): {e}")
        return jsonify({'error': 'Could not fetch users', 'details': str(e)}), 500

    finally:
        if conn:
            conn.close()

@app.route('/admin/rss-feed-categories', methods=['GET', 'POST'])
@login_required
def admin_rss_category_index():
    domain_configs = get_dynamic_db_configs()
    selected_db_key = request.args.get('selected_domain_id') or request.form.get('selected_domain_id')

    categories = []
    domain_label = "Unknown Domain"
    selected_domain_host = None  # Default fallback
    conn = None

    try:
        if selected_db_key:
            domain_info = domain_configs.get(selected_db_key)

            if not domain_info:
                flash("Selected domain not found in configuration.", "warning")
                return redirect(url_for('admin_dashboard'))

            numeric_domain_id = domain_info.get('id')
            domain_label = domain_info.get('label', selected_db_key)
            selected_domain_host = domain_info.get('host', 'localhost')

            print(f"📡 Filtering categories by domain ID: {numeric_domain_id} ({domain_label})")

            conn = get_local_db_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("""
                SELECT c.id, c.name, c.slug, COUNT(*) AS post_count
                FROM categories c
                JOIN local_generated_posts l ON l.category_id = c.id
                WHERE l.domain_id = %s
                GROUP BY c.id, c.name, c.slug
                ORDER BY c.name
            """, (numeric_domain_id,))
            categories = cursor.fetchall()
        else:
            flash("No domain selected. Please choose a domain to view its categories.", "info")

    except Exception as e:
        flash(f"❌ Error fetching categories: {e}", "danger")
        print(f"⚠️ Exception in admin_rss_category_index: {e}")
    finally:
        if conn:
            conn.close()

    return render_template(
        "admin_rss_category_list.html",
        categories=categories,
        selected_domain_id=selected_db_key,
        selected_domain_label=domain_label,
        selected_domain_host=selected_domain_host,
        domain_configs=domain_configs
    )

@app.route('/admin/au_campaigns', methods=['GET'])
@login_required  # Protect this route
def au_campaigns_view():
    """Displays a list of all AI campaigns with separate content and image generation logs."""
    conn = None
    campaigns = []
    content_logs = []
    image_logs = []

    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch campaign summaries
        cursor.execute("""
            SELECT id, csv_file_name, total_titles, success_count, fail_count, created_at, db_host_key
            FROM ai_campaigns
            ORDER BY created_at DESC
        """)
        campaigns = cursor.fetchall()

        # Fetch content generation logs
        cursor.execute("""
            SELECT blog_title, provider_name, model_name, status_code, error_message, created_at
            FROM blog_ai_error_logs
            WHERE log_type = 'content'
            ORDER BY created_at DESC
            LIMIT 100
        """)
        content_logs = cursor.fetchall()

        # Fetch image generation logs
        cursor.execute("""
            SELECT blog_title, provider_name, model_name, status_code, error_message, created_at
            FROM blog_ai_error_logs
            WHERE log_type = 'image'
            ORDER BY created_at DESC
            LIMIT 100
        """)
        image_logs = cursor.fetchall()

    except Exception as e:
        flash(f"Error fetching campaigns or logs: {str(e)}", "danger")
        print(f"Error fetching campaigns or logs: {e}")
    finally:
        if conn:
            conn.close()

    return render_template(
        'au_campaigns_table.html',
        campaigns=campaigns,
        content_logs=content_logs,
        image_logs=image_logs
    )

@app.route('/admin/blog_ai_error_logs', methods=['GET'])
@login_required # Protect this route
def blog_ai_error_logs_view():
    """Displays a list of all AI error logs."""
    conn = None
    error_logs = []
    try:
        conn = get_local_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, blog_title, provider_name, model_name, status_code, error_message, input_prompt, response, retry_attempt, created_at FROM blog_ai_error_logs ORDER BY created_at DESC")
        error_logs = cursor.fetchall()
    except Exception as e:
        flash(f"Error fetching error logs: {str(e)}", "danger")
        print(f"Error fetching error logs: {e}")
    finally:
        if conn:
            conn.close()
    return render_template('blog_ai_error_logs_table.html', error_logs=error_logs)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
