import os
import random
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
import spacy
import logging
import time
from flask import Flask, render_template, jsonify, send_file, request
from flask_socketio import SocketIO
from flask_cors import CORS
from html import escape
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging for model training
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Flask request logging to terminal
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
werkzeug_logger.addHandler(console_handler)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret')
CORS(app)
socketio = SocketIO(app)

# Configuration
DATABASE = 'expert_system.db'
DATASET_PATH = os.path.join('Combined Data.csv')
MODEL_FILE = 'mental_health_model.pkl'
RULES_PATH = os.path.join('scripts', 'rules.json')

# Load rules
try:
    with open(RULES_PATH, 'r') as f:
        rules = json.load(f)
except Exception as e:
    logger.error(f"Failed to load rules: {e}")
    rules = {"negative_keywords": [], "positive_keywords": [], "rules": {}}

# Load model
try:
    model = joblib.load(MODEL_FILE)
    if not hasattr(model, 'predict') or not isinstance(model.steps[0][1], TfidfVectorizer):
        logger.warning(f"Invalid model format in {MODEL_FILE}. Regenerating model on retraining.")
        model = None
except FileNotFoundError:
    logger.warning(f"Model file {MODEL_FILE} not found. Will create a new model on retraining.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}. Will create a new model on retraining.")
    model = None

# NLP pipelines
try:
    intent_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-base")
except Exception as e:
    logger.error(f"Intent classifier error: {e}")
    intent_classifier = lambda text, candidate_labels: {"labels": ["neutral"], "scores": [1.0]}

try:
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
except Exception as e:
    logger.error(f"Emotion classifier error: {e}")
    emotion_classifier = lambda text: [{"label": "sadness", "score": 0.0}, {"label": "anger", "score": 0.0}]

# Risk level coping strategies
coping_strategies = {
    "low": [
        {"text": "Take a deep breath", "type": "breathing"},
        {"text": "Try a 5-minute meditation", "type": "meditation"},
        {"text": "Drink some water", "type": "self-care"}
    ],
    "mild": [
        {"text": "Do a quick stretch", "type": "physical"},
        {"text": "Take a short walk", "type": "physical"},
        {"text": "List three things you’re grateful for (CBT: Gratitude Practice)", "type": "gratitude"}
    ],
    "medium": [
        {"text": "Challenge negative thoughts by writing down evidence for and against them (CBT: Cognitive Restructuring)", "type": "cbt-cognitive"},
        {"text": "Engage in a small, enjoyable activity (CBT: Behavioral Activation)", "type": "cbt-behavioral"},
        {"text": "Practice gratitude by noting three positive moments today (CBT: Gratitude Practice)", "type": "gratitude"}
    ],
    "high-risk": [
        {"text": "Consider professional help to explore CBT therapy", "type": "professional"},
        {"text": "Identify and challenge distorted thoughts with a trusted person (CBT: Cognitive Restructuring)", "type": "cbt-cognitive"},
        {"text": "Call a trusted person", "type": "social"}
    ],
    "extreme": [
        {"text": "Please contact a crisis counselor immediately", "type": "crisis"},
        {"text": "Call emergency services", "type": "crisis"},
        {"text": "Reach out to someone you trust", "type": "social"}
    ]
}

# Risk level logic
risk_level_from_score = lambda score: (
    "extreme" if score < 0.1 else
    "high-risk" if score < 0.2 else
    "medium" if score < 0.4 else
    "mild" if score <= 0.7 else
    "low"
)

# DB setup
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_history (
                        id INTEGER PRIMARY KEY, user_id TEXT, user_message TEXT, bot_response TEXT,
                        sentiment_score REAL, risk_level TEXT, strategy TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY, history_id INTEGER, feedback_score INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (history_id) REFERENCES user_history(id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY, user_id TEXT, strategy_type TEXT, preference_score INTEGER,
                        UNIQUE(user_id, strategy_type))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_surveys (
                        id INTEGER PRIMARY KEY, user_id TEXT, survey_type TEXT, score INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Add indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON user_history(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON user_history(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_id ON feedback(history_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_survey_user_id ON user_surveys(user_id)')
    
    # Check and add strategy column if missing
    cursor.execute("PRAGMA table_info(user_history)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'strategy' not in columns:
        logger.info("Adding 'strategy' column to user_history table")
        cursor.execute("ALTER TABLE user_history ADD COLUMN strategy TEXT")
    
    conn.commit()
    conn.close()

init_db()

# Session tracking for engagement metrics
session_start_times = {}
message_counts = {}

@app.before_request
def start_session_timer():
    if request.remote_addr not in session_start_times:
        session_start_times[request.remote_addr] = time.time()
        message_counts[request.remote_addr] = 0

def preprocess_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        df = pd.DataFrame({"text": [], "anxiety_label": []})
    non_anxious = ["I’m excited for the weekend!", "Work went well today."]
    df = pd.concat([df, pd.DataFrame({"text": non_anxious, "anxiety_label": [0]*len(non_anxious)})])
    df.to_csv(DATASET_PATH, index=False)
    return df

def analyze_sentiment_advanced(user_message):
    user_message = escape(user_message)
    sentiment_score = 0.5
    negative_count = sum(user_message.lower().count(k) for k in rules["negative_keywords"])
    positive_count = sum(user_message.lower().count(k) for k in rules.get("positive_keywords", []))
    try:
        emotions = emotion_classifier(user_message)
        if isinstance(emotions, list) and isinstance(emotions[0], list):
            emotions = emotions[0]
        sadness_score = next((e["score"] for e in emotions if e["label"] == "sadness"), 0.0)
        anger_score = next((e["score"] for e in emotions if e["label"] == "anger"), 0.0)
        joy_score = next((e["score"] for e in emotions if e["label"] == "joy"), 0.0)
        sentiment_score -= 0.2 * (sadness_score + anger_score)
        sentiment_score += 0.2 * joy_score + 0.1 * positive_count
    except Exception as e:
        logger.error(f"Emotion classifier error: {e}. Using default sentiment score.")
    sentiment_score = max(0.0, min(1.0, sentiment_score))
    return sentiment_score, negative_count

def lemmatize_text(text):
    if nlp:
        try:
            return ' '.join([token.lemma_ for token in nlp(text.lower())])
        except Exception as e:
            logger.error(f"Lemmatization error: {e}. Returning original text.")
    return text.lower()

def validate_message(message):
    if len(message.strip()) < 10:
        return False
    if len(set(message.lower().split())) < 3:
        return False
    return True

def collect_training_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''SELECT uh.user_message, uh.sentiment_score, uh.risk_level, f.feedback_score
                      FROM user_history uh
                      LEFT JOIN feedback f ON uh.id = f.history_id
                      WHERE uh.user_message IS NOT NULL''')
    rows = cursor.fetchall()
    conn.close()

    texts = []
    labels = []
    weights = []
    for row in rows:
        user_message, sentiment_score, risk_level, feedback_score = row
        if not validate_message(user_message):
            continue
        texts.append(user_message)
        risk_map = {"low": 0, "mild": 1, "medium": 2, "high-risk": 3, "extreme": 4}
        labels.append(risk_map.get(risk_level, 0))
        weights.append(feedback_score if feedback_score is not None else 3)

    return texts, labels, weights

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report: {json.dumps(report, indent=2)}")
    return accuracy

def retrain_model():
    texts, labels, weights = collect_training_data()
    if not texts or len(texts) < 20:
        logger.warning(f"Insufficient data for retraining: {len(texts)} samples")
        return

    global model
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        texts, labels, weights, test_size=0.2, random_state=42, stratify=labels
    )

    new_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
    new_model.fit(X_train, y_train, randomforestclassifier__sample_weight=w_train)

    new_accuracy = evaluate_model(new_model, X_test, y_test)

    if model:
        current_accuracy = evaluate_model(model, X_test, y_test)
        if new_accuracy > current_accuracy:
            model = new_model
            joblib.dump(model, MODEL_FILE)
            logger.info("New model saved: Improved accuracy")
        else:
            logger.info("New model discarded: No accuracy improvement")
    else:
        model = new_model
        joblib.dump(model, MODEL_FILE)
        logger.info("New model saved: No previous model")

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_model, 'interval', days=1)
scheduler.start()

def get_bot_response(user_message, user_id):
    user_message = escape(user_message)
    sentiment_score, negative_count = analyze_sentiment_advanced(user_message)
    risk_level = risk_level_from_score(sentiment_score)
    
    model_risk_level = None
    if model and hasattr(model, 'predict'):
        try:
            predicted_label = model.predict([user_message])[0]
            risk_map = {0: "low", 1: "mild", 2: "medium", 3: "high-risk", 4: "extreme"}
            model_risk_level = risk_map.get(predicted_label, risk_level)
            ordered = ["extreme", "high-risk", "medium", "mild", "low"]
            if ordered.index(model_risk_level) < ordered.index(risk_level):
                risk_level = model_risk_level
            logger.info(f"Prediction for user {user_id}: Model risk={model_risk_level}, Rule-based risk={risk_level}")
        except Exception as e:
            logger.error(f"Model prediction error: {e}. Falling back to rule-based risk level.")
    else:
        logger.warning("No valid model available. Using rule-based risk level.")

    try:
        intents = intent_classifier(user_message, candidate_labels=["venting", "seeking advice", "neutral"])
        primary_intent = intents["labels"][0]
    except Exception as e:
        logger.error(f"Intent classifier error: {e}. Defaulting to neutral intent.")
        primary_intent = "neutral"

    cbt_explanation = ""
    if risk_level in ["medium", "high-risk"]:
        cbt_explanation = " (CBT helps you identify and change negative thought patterns to improve your mood)"

    strategy = None
    strategy_text = None
    for rule, details in rules["rules"].items():
        if lemmatize_text(rule) in lemmatize_text(user_message):
            rule_level = details["level"]
            ordered = ["extreme", "high-risk", "medium", "mild", "low"]
            if ordered.index(rule_level) < ordered.index(risk_level):
                risk_level = rule_level
                if "strategy" in details:
                    strategy = {"text": details["strategy"], "type": details.get("type", "general")}
                    strategy_text = strategy["text"]
                    if risk_level in ["medium", "high-risk"] and "cbt" in strategy["type"]:
                        strategy_text += cbt_explanation
                    break
    else:
        preferences = {}
        feedback_scores = {}
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('SELECT strategy_type, preference_score FROM user_preferences WHERE user_id = ?', (user_id,))
            preferences = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Check if strategy column exists
            cursor.execute("PRAGMA table_info(user_history)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'strategy' in columns:
                cursor.execute('''SELECT uh.strategy, AVG(f.feedback_score) as avg_score
                                 FROM user_history uh
                                 LEFT JOIN feedback f ON uh.id = f.history_id
                                 WHERE uh.user_id = ? AND uh.strategy IS NOT NULL
                                 GROUP BY uh.strategy''', (user_id,))
                feedback_scores = {row[0]: row[1] for row in cursor.fetchall() if row[1] is not None}
            else:
                logger.warning("Strategy column missing in user_history. Skipping feedback scores.")
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}. Using default strategy selection.")

        available_strategies = coping_strategies.get(risk_level, coping_strategies["low"])
        scored_strategies = []
        for s in available_strategies:
            score = preferences.get(s["type"], 3)
            feedback_score = feedback_scores.get(s["text"], 3)
            total_score = (score * 0.6) + (feedback_score * 0.4)
            scored_strategies.append((s, total_score))
        if scored_strategies:
            strategy = max(scored_strategies, key=lambda x: x[1])[0]
        else:
            strategy = random.choice(available_strategies)
        strategy_text = strategy["text"]
        if risk_level in ["medium", "high-risk"] and "cbt" in strategy["type"]:
            strategy_text += cbt_explanation

    response = f"{strategy_text}" if primary_intent != "seeking advice" else f"{strategy_text}. Would you like resources to learn more?"

    history_id = None
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO user_history (user_id, user_message, bot_response, sentiment_score, risk_level, strategy)
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (user_id, user_message, response, sentiment_score, risk_level, strategy["text"]))
        conn.commit()
        history_id = cursor.lastrowid
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Database insertion error: {e}")
        return response, sentiment_score, None

    return response, sentiment_score, history_id

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join('static', 'favicon.ico'), mimetype='image/x-icon')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    user_id = request.remote_addr
    message_counts[user_id] += 1
    try:
        response, score, history_id = get_bot_response(user_message, user_id)
        if history_id is None:
            logger.error("Failed to store message in database.")
            return jsonify({"response": "Failed to store message in database."}), 500
        return jsonify({"response": response, "sentiment_score": score, "history_id": history_id})
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
        print(f"Error in get_response: {e}")
        return jsonify({"response": "An error occurred while processing your message."}), 500

@app.route('/resources')
def resources():
    risk_level = request.args.get('risk_level', 'low')
    return jsonify({
        "low": [{"title": "Mindfulness", "url": "https://www.mindful.org"}],
        "mild": [{"title": "Self-care tips", "url": "https://www.mentalhealth.org.uk"}],
        "medium": [{"title": "Talk to someone", "url": "https://www.betterhelp.com"}],
        "high-risk": [{"title": "Crisis Help", "url": "https://988lifeline.org"}],
        "extreme": [{"title": "Emergency Help", "url": "https://www.opencounseling.com/suicide-hotlines"}]
    }.get(risk_level, []))

@app.route('/trends')
def trends():
    user_id = request.remote_addr
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT sentiment_score, timestamp FROM user_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    scores = [row[0] for row in rows]
    timestamps = [row[1] for row in rows]
    return jsonify({"sentiment_scores": scores, "timestamps": timestamps})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    score = int(data.get('score', 0))
    history_id = int(data.get('history_id', 0))
    if not (1 <= score <= 5):
        return jsonify({"message": "Score must be between 1 and 5"}), 400
    if not history_id:
        return jsonify({"message": "Invalid history ID"}), 400

    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedback (history_id, feedback_score) VALUES (?, ?)',
                       (history_id, score))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return jsonify({"message": "Error saving feedback"}), 500
    finally:
        conn.close()

    return jsonify({"message": "Feedback submitted successfully"})

@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    data = request.get_json()
    user_id = request.remote_addr
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        for strategy_type, score in data.items():
            cursor.execute('''INSERT OR REPLACE INTO user_preferences (user_id, strategy_type, preference_score)
                             VALUES (?, ?, ?)''', (user_id, strategy_type, int(score)))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error saving preferences: {e}")
        return jsonify({"message": "Error saving preferences"}), 500
    finally:
        conn.close()
    return jsonify({"message": "Preferences saved successfully"})

@app.route('/save_survey', methods=['POST'])
def save_survey():
    data = request.get_json()
    user_id = request.remote_addr
    score = data.get('score')
    survey_type = data.get('survey_type')
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO user_surveys (user_id, survey_type, score)
                         VALUES (?, ?, ?)''', (user_id, survey_type, score))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error saving survey: {e}")
        return jsonify({"message": "Error saving survey"}), 500
    finally:
        conn.close()
    return jsonify({"message": "Survey saved successfully"})

@app.route('/survey_results')
def survey_results():
    user_id = request.remote_addr
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT survey_type, score, timestamp
                         FROM user_surveys
                         WHERE user_id = ?
                         ORDER BY timestamp DESC''', (user_id,))
        surveys = [{"type": row[0], "score": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
        conn.close()
        pre_scores = [s["score"] for s in surveys if s["type"] == "pre"]
        post_scores = [s["score"] for s in surveys if s["type"] == "post"]
        avg_pre = sum(pre_scores) / len(pre_scores) if pre_scores else 0
        avg_post = sum(post_scores) / len(post_scores) if post_scores else 0
        return jsonify({
            "surveys": surveys,
            "avg_pre_score": avg_pre,
            "avg_post_score": avg_post,
            "improvement": avg_pre - avg_post if avg_pre and avg_post else 0
        })
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving survey results: {e}")
        return jsonify({"message": "Error retrieving survey results"}), 500

@app.route('/engagement_metrics')
def engagement_metrics():
    user_id = request.remote_addr
    session_duration = time.time() - session_start_times.get(user_id, time.time())
    message_count = message_counts.get(user_id, 0)
    return jsonify({
        "session_duration_seconds": session_duration,
        "message_count": message_count
    })

if __name__ == '__main__':
    socketio.run(app, host='192.168.8.15', port=5000, debug=True)