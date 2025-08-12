import logging
for name in list(logging.root.manager.loggerDict):
    if name.startswith("streamlit"):
        logging.getLogger(name).setLevel(logging.ERROR)

import streamlit as st
import sqlite3, pandas as pd, requests

# Streamlit UI
st.title("🏨 Hotel ChatBot with Local SQLCoder")

# Connect to local SQLite DB
conn = sqlite3.connect("hotel_operations.db")

# Generate SQL via SQLCoder
def get_sql(nl_query: str) -> str:
    resp = requests.post("http://localhost:8000/query", json={"question": nl_query})
    return resp.json().get("sql", "-- Error: no SQL returned")

# User input
user_query = st.text_input("Ask about hotel operations:")

if user_query:
    sql = get_sql(user_query)
    st.subheader("Generated SQL")
    st.code(sql, language="sql")
    try:
        df = pd.read_sql(sql, conn)
        st.subheader("Result")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Query failed: {e}")
