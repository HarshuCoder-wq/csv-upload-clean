import csv
from io import StringIO
from uploader.question_ai_handler import get_answer_from_ai
from config.db_config import get_db_connection

def process_csv_and_answer(file_stream, db_key=None):
    conn = get_db_connection(db_key) if db_key else get_db_connection()
    cursor = conn.cursor(dictionary=True)

    reader = csv.DictReader(StringIO(file_stream.read().decode('utf-8')))
    questions = list(reader)

    # Get all user_ids from the users table
    cursor.execute("SELECT id FROM users")
    all_users = sorted([str(row['id']) for row in cursor.fetchall()])  # type: ignore # Sort for consistency

    responses = []

    for row in questions:
        question_text = row.get('question', '').strip()
        user_id = row.get('user_id', '').strip()

        if not question_text or not user_id:
            continue  # Skip invalid rows

        try:
            user_id = int(user_id)
        except ValueError:
            continue  # Skip if user_id is not a valid int

        # Insert question into DB
        cursor.execute("""
            INSERT INTO questions (question_text, user_id, created_by, updated_by)
            VALUES (%s, %s, %s, %s)
        """, (question_text, user_id, user_id, user_id))
        question_id = cursor.lastrowid

        answers_for_this_question = []

        for uid in all_users:
            try:
                uid = int(uid)
            except ValueError:
                continue

            if uid == user_id:
                continue  # Skip answering user's own question

            # Generate answer from AI
            answer = get_answer_from_ai(question_text, for_user_id=uid)

            cursor.execute("""
                INSERT INTO answers (question_id, user_id, answer_text, created_by, updated_by)
                VALUES (%s, %s, %s, %s, %s)
            """, (question_id, uid, answer, uid, uid))

            answers_for_this_question.append({
                'user_id': uid,
                'answer': answer
            })

        responses.append({
            'question_id': question_id,
            'question': question_text,
            'excluded_user': user_id,
            'answers': answers_for_this_question
        })

    conn.commit()
    cursor.close()
    conn.close()

    return responses
