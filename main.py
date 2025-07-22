import streamlit as st
import sqlite3
import pandas as pd
import torch
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─── Load model & tokenizer once ─────────────────────────────────────────────
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

# ─── Initialize SQLite DB ────────────────────────────────────────────────────
def initialize_database(conn, sql_file_path="bulk_hotel_data_realistic.sql"):
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()

db_path = "bulk_hotel_data_realistic.db"
conn = sqlite3.connect(db_path)

if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regions';").fetchone():
    initialize_database(conn, "bulk_hotel_data_realistic.sql")

# ─── Generate schema dynamically from DB ─────────────────────────────────────
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
    prompt = f"""### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
{schema_text}

### SQL
"""
    return prompt

# ─── Prompt helpers ──────────────────────────────────────────────────────────
def clean_sql(raw_sql: str) -> str:
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

def nl_to_sql(question: str) -> str:
    prompt = get_dynamic_schema_prompt(conn, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sql(raw)

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
st.caption(f"🚀 Model loaded in **{model_load_latency:.2f} seconds**")

# ─── User Query Area ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")
with st.expander("Example: 'Which hotel had the most bookings in June?'", expanded=False):
    pass

user_query = st.text_input("Ask about hotel operations:", placeholder="E.g. How many guests stayed in Paris?")

if user_query:
    try:
        sql = nl_to_sql(user_query)
        if not sql.strip():
            raise ValueError("The generated SQL query is empty.")

        st.markdown("#### 🔧 Generated SQL Query")
        st.code(sql, language="sql")

        try:
            start_time = time.time()
            df = pd.read_sql(sql, conn)
            end_time = time.time()
            latency = end_time - start_time

            st.success("Query executed successfully.")
            st.markdown("#### 📊 Result")
            st.dataframe(df, use_container_width=True)
            st.caption(f"⏱️ Query time: **{latency:.4f} seconds**")
        except Exception as sql_error:
            st.error("❌ Something went wrong while executing your query.")
            with st.expander("Show full error"):
                st.code(str(sql_error), language="text")


    except Exception as gen_error:
        st.error(f"❌ Failed to generate a valid SQL query:\n{gen_error}")


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