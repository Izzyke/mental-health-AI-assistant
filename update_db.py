import sqlite3

DATABASE = 'expert_system.db'

def update_db_schema():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if strategy column exists
    cursor.execute("PRAGMA table_info(user_history)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'strategy' not in columns:
        print("Adding 'strategy' column to user_history table...")
        cursor.execute("ALTER TABLE user_history ADD COLUMN strategy TEXT")
        conn.commit()
        print("Column added successfully.")
    else:
        print("'strategy' column already exists.")
    
    conn.close()

if __name__ == "__main__":
    update_db_schema()