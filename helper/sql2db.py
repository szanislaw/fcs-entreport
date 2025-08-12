import sqlite3

# File paths
sql_file = 'job_detail_listing.sql'
db_file = 'job_detail_listing.db'

# Create SQLite connection
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Read SQL content
with open(sql_file, 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Execute SQL (handles CREATE and INSERT statements)
try:
    cursor.executescript(sql_script)
    conn.commit()
    print(f"✅ Database created successfully: {db_file}")
except Exception as e:
    print(f"❌ Error occurred: {e}")

conn.close()
