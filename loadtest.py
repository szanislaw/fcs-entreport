import random
import time
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# ─── Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
model = AutoModelForCausalLM.from_pretrained(
    "defog/sqlcoder-7b-2",
    torch_dtype=torch.float16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded on {device}")

# ─── Sample queries ─────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "How many rooms were cleaned yesterday?",
    "Show me top 3 staff by rooms cleaned.",
    "How many guest complaints this month?",
    "What is the inspection pass rate?",
    "List bookings in July.",
    "Average cleaning time per room?",
    "Which staff worked overtime most?",
    "Number of complaints by room?",
    "Rooms needing recleaning this week?",
    "Who cleaned the most rooms last month?"
]

# ─── SQLite connection ───────────────────────────────────────────────────────
conn = sqlite3.connect("hotel.db", check_same_thread=False)

# ─── PG to SQLite cleaner ────────────────────────────────────────────────────
def pg_to_sqlite(sql: str) -> str:
    cleaned = sql
    cleaned = re.sub(r'\bpublic\.', '', cleaned)
    cleaned = re.sub(r'\bTRUE\b', '1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bFALSE\b', '0', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'::\s*\w+', '', cleaned)
    cleaned = re.sub(r'timestamp(?:\(\d+\))?\s+without time zone', 'datetime', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bSERIAL\b', 'INTEGER', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'USING\s+btree', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bRETURNING\b.*?(;|\n|$)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip().rstrip(';')
    cleaned = re.sub(
        r'\bAVG\s*\(\s*([a-zA-Z_][\w\.]*)\s*-\s*([a-zA-Z_][\w\.]*)\s*\)',
        r'ROUND(AVG((julianday(\1) - julianday(\2)) * 24 * 60), 2)',
        cleaned
    )
    return cleaned

# ─── Get schema for prompt ───────────────────────────────────────────────────
def get_dynamic_schema_prompt(question: str) -> str:
    schema = []
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_defs = [f"{col[1]} {col[2]}" for col in columns]
        schema.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")

    schema_text = "\n".join(schema)
    prompt = f"""### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
{schema_text}

### SQL
"""
    return prompt

# ─── Extract SQL ─────────────────────────────────────────────────────────────
def clean_sql(raw_sql: str) -> str:
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

# ─── NL to SQL ───────────────────────────────────────────────────────────────
def nl_to_sql(question: str) -> str:
    prompt = get_dynamic_schema_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = clean_sql(raw)
    return pg_to_sqlite(sql)

# ─── Run 1 Task ──────────────────────────────────────────────────────────────
def run_test(i):
    q = random.choice(SAMPLE_QUERIES)
    print(f"[{i}] Testing: {q}")
    try:
        start_time = time.time()
        sql = nl_to_sql(q)
        inference_time = time.time() - start_time
        print(f"[{i}] SQL: {sql}")
        
        # Execute the generated SQL
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        print(f"[{i}] ✅ Success | {len(rows)} rows | ⏱️ {inference_time:.2f}s")
    except Exception as e:
        print(f"[{i}] ❌ Error: {e}")

# ─── Multi-threaded Load Test ────────────────────────────────────────────────
if __name__ == "__main__":
    num_tasks = 500   # Change to 100 or more for high load
    with ThreadPoolExecutor(max_workers=8) as executor:  # Increase workers for more load
        futures = [executor.submit(run_test, i) for i in range(num_tasks)]
        for _ in as_completed(futures):
            pass
