import sqlite3
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Configuration
DATABASE = 'expert_system.db'
DATASET_PATH = '/home/israel/Documents/Combined Data.csv'
MODEL_FILE = 'mental_health_model.pkl'
MIN_RECORDS = 10  # Minimum records required from user_history

# Synthetic data for bootstrapping (if database and dataset are insufficient)
SYNTHETIC_DATA = [
    {"text": "I’m so happy today!", "risk_level": "low"},
    {"text": "Feeling great about my work.", "risk_level": "low"},
    {"text": "I’m feeling sad today.", "risk_level": "mild"},
    {"text": "Really anxious right now.", "risk_level": "medium"},
    {"text": "Feeling hopeless.", "risk_level": "high-risk"},
    {"text": "I can’t go on.", "risk_level": "extreme"},
    {"text": "Just chilling.", "risk_level": "low"},
    {"text": "Work is stressful.", "risk_level": "medium"},
    {"text": "I’m excited for the weekend!", "risk_level": "low"},
    {"text": "Feeling lonely.", "risk_level": "medium"}
]

def collect_training_data():
    """Collect data from user_history and feedback tables."""
    texts = []
    labels = []
    weights = []
    
    # Query user_history and feedback
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT uh.user_message, uh.risk_level, f.feedback_score
                          FROM user_history uh
                          LEFT JOIN feedback f ON uh.id = f.history_id
                          WHERE uh.user_message IS NOT NULL''')
        rows = cursor.fetchall()
        conn.close()

        risk_map = {"low": 0, "mild": 1, "medium": 2, "high-risk": 3, "extreme": 4}
        for row in rows:
            user_message, risk_level, feedback_score = row
            texts.append(user_message)
            labels.append(risk_map.get(risk_level, 0))
            weights.append(feedback_score if feedback_score is not None else 3)
        
        print(f"Collected {len(texts)} records from user_history.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    
    return texts, labels, weights

def load_dataset():
    """Load data from Combined Data.csv or return empty DataFrame."""
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {len(df)} records from {DATASET_PATH}.")
        # Assume columns: 'text' (message) and 'anxiety_label' (0 for low, 1 for high)
        # Map anxiety_label to risk levels (simplified)
        risk_map = {0: "low", 1: "high-risk"}
        texts = df['text'].tolist()
        labels = [risk_map.get(label, "low") for label in df['anxiety_label']]
        weights = [3] * len(texts)  # Default weight
        return texts, labels, weights
    except FileNotFoundError:
        print(f"Dataset file {DATASET_PATH} not found.")
        return [], [], []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], [], []

def train_new_model():
    """Train a new model using available data."""
    # Collect data from database
    texts, labels, weights = collect_training_data()
    
    # If insufficient data, try dataset
    if len(texts) < MIN_RECORDS:
        print(f"Insufficient database records ({len(texts)}). Trying dataset.")
        dataset_texts, dataset_labels, dataset_weights = load_dataset()
        texts.extend(dataset_texts)
        labels.extend([{"low": 0, "high-risk": 3}.get(label, 0) for label in dataset_labels])
        weights.extend(dataset_weights)
    
    # If still insufficient, use synthetic data
    if len(texts) < MIN_RECORDS:
        print(f"Insufficient dataset records. Using synthetic data.")
        synthetic_texts = [item["text"] for item in SYNTHETIC_DATA]
        synthetic_labels = [{"low": 0, "mild": 1, "medium": 2, "high-risk": 3, "extreme": 4}[item["risk_level"]] for item in SYNTHETIC_DATA]
        synthetic_weights = [3] * len(synthetic_texts)
        texts.extend(synthetic_texts)
        labels.extend(synthetic_labels)
        weights.extend(synthetic_weights)
    
    if len(texts) < MIN_RECORDS:
        print(f"Still insufficient data ({len(texts)} records). Need at least {MIN_RECORDS}.")
        return False
    
    # Train model
    print(f"Training model with {len(texts)} records.")
    model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
    model.fit(texts, labels, randomforestclassifier__sample_weight=weights)
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}.")
    return True

if __name__ == "__main__":
    success = train_new_model()
    if not success:
        print("Failed to train model. Please add more data to user_history or ensure Combined Data.csv is available.")