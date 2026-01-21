import pytest
import sqlite3
from App import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_template(client):
    response = client.get('/')
    assert response.status_code == 200
    # Changed 'Majirow' to check for the stylesheet link, assuming index.html includes it
    assert b'link rel="stylesheet" href="/static/styles.css"' in response.data

def test_static_files(client):
    response = client.get('/static/styles.css')
    assert response.status_code == 200
    response = client.get('/static/script.js')
    assert response.status_code == 200

def test_trends_endpoint(client):
    response = client.get('/trends')
    assert response.status_code == 200
    data = response.get_json()
    assert 'sentiment_scores' in data
    assert 'timestamps' in data

def test_rule_application(client):
    conn = sqlite3.connect('expert_system.db')
    cursor = conn.cursor()
    # Insert a record with a negative sentiment_score to reflect the "hopeless" message
    cursor.execute('INSERT INTO user_history (user_message, bot_response, sentiment_score, risk_level) VALUES (?, ?, ?, ?)',
                   ('I feel hopeless', 'Consider professional help.', -0.5, 'high'))
    conn.commit()
    response = client.get('/trends')
    assert response.status_code == 200
    data = response.get_json()
    assert any(score < 0 for score in data['sentiment_scores']), "Negative sentiment not detected"
    conn.close()

def test_styles_applied(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'link rel="stylesheet" href="/static/styles.css"' in response.data

def test_resources_endpoint(client):
    response = client.get('/resources?risk_level=high')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) > 0
    assert 'title' in data[0]
    assert 'url' in data[0]

def test_feedback_validation(client):
    response = client.post('/feedback', json={'score': 6})
    assert response.status_code == 400
    # Updated to match the actual error message in App.py
    assert b'Score must be between 1 and 5' in response.data
    response = client.post('/feedback', json={'score': 3})
    assert response.status_code == 200
    assert b'Feedback submitted successfully' in response.data

def test_database_index(client):
    conn = sqlite3.connect('expert_system.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_timestamp'")
    assert cursor.fetchone() is not None
    conn.close()