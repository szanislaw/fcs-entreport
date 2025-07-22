import streamlit as st
import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# ─── Load model & tokenizer once ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    # tokenizer stays the same
    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")

    # load in fp16 for speed, then move to cuda
    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True       # trims CPU memory usage
    )
    # send to GPU
    model = model.to("cuda:0")

    # enable cuDNN auto-tuner for optimal kernels
    torch.backends.cudnn.benchmark = True

    return tokenizer, model

tokenizer, model = load_model()

# ─── Prompt schema setup ───────────────────────────────────────────────────────
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
    """Extract SQL starting from first keyword to avoid prompt leftovers."""
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
        do_sample=True,       # Sampling is more MPS-compatible
        temperature=0.7,
        top_p=0.9
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_sql(raw)

# ─── Initialize SQLite DB ────────────────────────────────────────────────────
conn = sqlite3.connect("hotel_operations.db")

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.title("🏨 Hotel ChatBot with Local SQLCoder on MPS")

user_query = st.text_input("Ask about hotel operations:")

if user_query:
    sql = nl_to_sql(user_query)
    st.subheader("🔧 Generated SQL")
    st.code(sql, language="sql")

    try:
        df = pd.read_sql(sql, conn)
        st.subheader("📊 Result")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to execute SQL:\n{e}")
