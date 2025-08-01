import sqlite3
import pandas as pd
import os

# Set your paths
csv_folder = "csvs"  # folder containing all 15 CSV files
schema_file = "cleaning.sql"
db_file = "cleaning.db"

# Connect to SQLite DB
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Step 1: Create tables from schema
with open(schema_file, "r") as f:
    schema_sql = f.read()
cursor.executescript(schema_sql)
conn.commit()

# Step 2: Load each CSV and insert into the corresponding table
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        table_name = file.replace(".csv", "")
        file_path = os.path.join(csv_folder, file)

        print(f"Inserting into {table_name}...")

        df = pd.read_csv(file_path)

        # Clean column names to match schema (strip extra spaces, if any)
        df.columns = [col.strip() for col in df.columns]

        # Insert into SQLite (if table exists)
        df.to_sql(table_name, conn, if_exists='append', index=False)

print("✅ All tables populated.")
conn.close()
