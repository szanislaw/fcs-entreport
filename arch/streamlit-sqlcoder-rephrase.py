import streamlit as st
import sqlite3
import pandas as pd
import torch
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer as HFTokenizer, AutoModelForSeq2SeqLM  # for rephrasing

# ─── Load SQLCoder Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model_with_timing():
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        quantization_config=bnb_config,
        device_map="auto"
    )

    end_time = time.time()
    latency = end_time - start_time
    return tokenizer, model, latency

tokenizer, model, model_load_latency = load_model_with_timing()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load Rephraser Model ────────────────────────────────────────────────────
@st.cache_resource
def load_rephraser():
    start_time = time.time()

    tokenizer = HFTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    end_time = time.time()
    latency = end_time - start_time
    return tokenizer, model, latency

rephraser_tokenizer, rephraser_model, rephraser_load_latency = load_rephraser()

# ─── SQLite Setup ────────────────────────────────────────────────────────────
def initialize_database(conn, sql_file_path="bulk_hotel_data_realistic.sql"):
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()

db_path = "bulk_hotel_data_realistic.db"
conn = sqlite3.connect(db_path)

if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regions';").fetchone():
    initialize_database(conn, "bulk_hotel_data_realistic.sql")

# ─── Schema Prompt ───────────────────────────────────────────────────────────
def get_dynamic_schema_prompt(conn: sqlite3.Connection, question: str) -> str:
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
    prompt = f"""
### Task
Generate an accurate and executable SQL query based on the question.
Use explicit conditions and avoid ambiguity.

### Example:
Q: Which hotel had the most guests in June?
A:
SELECT h.hotel_name, COUNT(*) AS guest_count
FROM bookings b
JOIN rooms r ON b.room_id = r.room_id
JOIN hotels h ON r.hotel_id = h.hotel_id
WHERE strftime('%m', b.check_in) = '06'
GROUP BY h.hotel_id
ORDER BY guest_count DESC
LIMIT 1;

### Question:
{question}

### Database Schema
{schema_text}

### SQL
"""
    return prompt

# ─── Rephrase Question ───────────────────────────────────────────────────────
def rephrase_question_llm(question: str) -> tuple[str, float]:
    start_time = time.time()
    prompt = f"Rephrase this question to be explicit for SQL: {question}"
    inputs = rephraser_tokenizer(prompt, return_tensors="pt", truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    outputs = rephraser_model.generate(**inputs, max_new_tokens=64)
    rephrased = rephraser_tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    latency = end_time - start_time
    return rephrased, latency

# ─── SQL Generation ──────────────────────────────────────────────────────────
def clean_sql(raw_sql: str) -> str:
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

def nl_to_sql(question: str) -> tuple[str, str, float]:
    rephrased, rephrase_latency = rephrase_question_llm(question)
    prompt = get_dynamic_schema_prompt(conn, rephrased)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    end_time = time.time()
    gen_latency = end_time - start_time

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sql(raw), rephrased, rephrase_latency + gen_latency

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel Manager Dashboard", page_icon="🏨", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 Hotel Manager Dashboard")
st.caption(f"🚀 SQLCoder loaded in **{model_load_latency:.2f} sec**, Rephraser in **{rephraser_load_latency:.2f} sec**")

# ─── Query Area ──────────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")
with st.expander("Example: 'Which hotel had the most bookings in June?'", expanded=False):
    pass

user_query = st.text_input("Ask about hotel operations:", placeholder="E.g. How many guests stayed in Paris?")

if user_query:
    sql, clarified, total_latency = nl_to_sql(user_query)
    st.markdown("#### 📝 Rephrased Question")
    st.info(clarified)

    st.markdown("#### 🔧 Generated SQL Query")
    st.code(sql, language="sql")
    st.caption(f"🧠 Inference time: **{total_latency:.2f} sec**")

    try:
        start_time = time.time()
        df = pd.read_sql(sql, conn)
        end_time = time.time()
        latency = end_time - start_time

        st.success("Query executed successfully.")
        st.markdown("#### 📊 Result")
        st.dataframe(df, use_container_width=True)
        st.caption(f"⏱️ Query time: **{latency:.4f} seconds**")
    except Exception as e:
        st.error(f"❌ SQL Error: {e}")

# ─── Dashboard Section ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Hotel Performance Overview")

col1, col2 = st.columns(2)

dashboard_queries = {
    "💰 Most Profitable Room": """
        SELECT r.room_id, r.room_type, h.hotel_name, SUM(b.total_price) AS total_revenue
        FROM bookings b
        JOIN rooms r ON b.room_id = r.room_id
        JOIN hotels h ON r.hotel_id = h.hotel_id
        GROUP BY r.room_id
        ORDER BY total_revenue DESC
        LIMIT 1;
    """,
    "🏨 Most Profitable Hotel": """
        SELECT h.hotel_name, SUM(b.total_price) AS total_revenue
        FROM bookings b
        JOIN rooms r ON b.room_id = r.room_id
        JOIN hotels h ON r.hotel_id = h.hotel_id
        GROUP BY h.hotel_id
        ORDER BY total_revenue DESC
        LIMIT 1;
    """,
    "🌎 Top Guest Nationalities": """
        SELECT g.nationality, COUNT(*) AS num_bookings
        FROM bookings b
        JOIN guests g ON b.guest_id = g.guest_id
        GROUP BY g.nationality
        ORDER BY num_bookings DESC
        LIMIT 10;
    """,
    "📅 Average Stay Duration": """
        SELECT h.hotel_name,
               ROUND(AVG(julianday(b.check_out) - julianday(b.check_in)), 2) AS avg_stay_days
        FROM bookings b
        JOIN rooms r ON b.room_id = r.room_id
        JOIN hotels h ON r.hotel_id = h.hotel_id
        GROUP BY h.hotel_id
        ORDER BY avg_stay_days DESC
        LIMIT 10;
    """
}

for idx, (title, query) in enumerate(dashboard_queries.items()):
    with (col1 if idx % 2 == 0 else col2):
        st.markdown(f"### {title}")
        try:
            start_time = time.time()
            df_result = pd.read_sql(query, conn)
            end_time = time.time()
            latency = end_time - start_time

            st.dataframe(df_result, use_container_width=True)
            st.caption(f"⏱️ Loaded in {latency:.4f} sec")
        except Exception as e:
            st.error(f"Failed to load {title}: {e}")
