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
def initialize_database(conn, sql_file_path="hotel_schema.sql"):
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()

db_path = "hotel_operations.db"
conn = sqlite3.connect(db_path)

# Only initialize once if not present
if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regions';").fetchone():
    initialize_database(conn, "/home/shawnyzy/Documents/GitHub/fcs-entreport/hotel_operations.sql")

# ─── Prompt schema setup ─────────────────────────────────────────────────────
schema_prompt = """### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name TEXT);
CREATE TABLE hotels (hotel_id INTEGER PRIMARY KEY, region_id INTEGER, hotel_name TEXT, city TEXT);
CREATE TABLE rooms (room_id INTEGER PRIMARY KEY, hotel_id INTEGER, room_type TEXT, base_price REAL);
CREATE TABLE guests (guest_id INTEGER PRIMARY KEY, guest_name TEXT, nationality TEXT);
CREATE TABLE bookings (booking_id INTEGER PRIMARY KEY, room_id INTEGER, check_in TEXT, check_out TEXT, guest_id INTEGER, total_price REAL);

### SQL
"""

def clean_sql(raw_sql: str) -> str:
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

def nl_to_sql(question: str) -> str:
    prompt = schema_prompt.format(question=question)
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

# ─── Streamlit UI ───────────────────────────────────────────────────────────
st.title("🏨 Hotel ChatBot with Local SQLCoder on CUDA (4-bit)")
st.caption(f"🚀 Model loaded in **{model_load_latency:.2f} seconds**")


user_query = st.text_input("Ask about hotel operations (e.g., 'How many regions are there?'):")

if user_query:
    sql = nl_to_sql(user_query)
    st.subheader("🔧 Generated SQL")
    st.code(sql, language="sql")

    try:
        start_time = time.time()
        df = pd.read_sql(sql, conn)
        end_time = time.time()

        latency = end_time - start_time
        st.subheader("📊 Result")
        st.dataframe(df)
        st.caption(f"⏱️ Query executed in **{latency:.6f} seconds**")
    except Exception as e:
        st.error(f"❌ Failed to execute SQL:\n{e}")

        
st.markdown("---")
st.header("📈 Hotel Operations Dashboard")

# Define dashboard queries
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
    "🌎 Bookings by Nationality": """
        SELECT g.nationality, COUNT(*) AS num_bookings
        FROM bookings b
        JOIN guests g ON b.guest_id = g.guest_id
        GROUP BY g.nationality
        ORDER BY num_bookings DESC
        LIMIT 10;
    """,
    "📅 Average Stay Duration by Hotel": """
        SELECT h.hotel_name,
               AVG(julianday(b.check_out) - julianday(b.check_in)) AS avg_stay
        FROM bookings b
        JOIN rooms r ON b.room_id = r.room_id
        JOIN hotels h ON r.hotel_id = h.hotel_id
        GROUP BY h.hotel_id
        ORDER BY avg_stay DESC
        LIMIT 10;
    """
}

# Display results for each dashboard query
for title, query in dashboard_queries.items():
    st.subheader(title)
    try:
        start_time = time.time()
        df_result = pd.read_sql(query, conn)
        end_time = time.time()
        latency = end_time - start_time

        st.dataframe(df_result)
        st.caption(f"⏱️ Query executed in **{latency:.6f} seconds**")
    except Exception as e:
        st.error(f"Error loading '{title}': {e}")


