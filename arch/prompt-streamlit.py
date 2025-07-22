import streamlit as st
import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -- Load model and tokenizer once --
@st.cache_resource
def load_model():
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", load_in_4bit=True, device_map="auto")

    return tokenizer, model

tokenizer, model = load_model()

# -- Prompt template with embedded schema --
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
# -- Function to generate SQL from natural language --
def nl_to_sql(question: str) -> str:
    prompt = schema_prompt.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, do_sample=False)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql.strip()

# -- Initialize SQLite --
conn = sqlite3.connect("hotel_operations.db")

# -- Streamlit UI --
st.title("🏨 Hotel ChatBot with Local SQLCoder (Transformer-based)")

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
