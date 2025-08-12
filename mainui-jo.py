import re
import streamlit as st
import sqlite3
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sql_fixes import apply_sql_fixes 

import os

# # ─────────────────────────────────────────────────────────────────────────────
# # Optional: Build .db from .sql file
# # ─────────────────────────────────────────────────────────────────────────────
# def build_db_from_sql(sql_file: str, db_file: str):
#     if not os.path.exists(db_file) and os.path.exists(sql_file):
#         with open(sql_file, 'r', encoding='utf-8') as f:
#             sql_script = f.read()
#         conn = sqlite3.connect(db_file)
#         try:
#             conn.executescript(sql_script)
#             print(f"✅ Built database '{db_file}' from '{sql_file}'")
#         except Exception as e:
#             print(f"❌ Failed to build DB: {e}")
#         finally:
#             conn.close()

# # Path settings
# SQL_PATH = "schemas/job_detail_listing.sql"
# DB_PATH = "schemas/job_detail_listing.db"

# # Build database from .sql file if needed
# build_db_from_sql(SQL_PATH, DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Layout Setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 90rem;
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load SQLCoder model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    end = time.time()
    return tokenizer, model, end - start

tokenizer, model, model_latency = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conn = sqlite3.connect("schemas/job_detail_listing.db")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: prompt builder and SQL cleaner
# ─────────────────────────────────────────────────────────────────────────────
def get_schema_prompt(conn, question):
    cursor = conn.cursor()
    schema = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_defs = [f"{col[1]} {col[2]}" for col in columns]
        schema.append(f"CREATE TABLE {table} ({', '.join(col_defs)});")
    schema_text = "\n".join(schema)

    # Load system prompt from external file
    with open("prompt/jo-sysprompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(question=question, schema_text=schema_text)
    print(prompt)
    return prompt


def clean_sql(raw_sql):
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

def nl_to_sql(question):
    prompt = get_schema_prompt(conn, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = clean_sql(raw)

    # Apply all known fixes
    sql = apply_sql_fixes(sql)

    return sql


# ─────────────────────────────────────────────────────────────────────────────
# Title and query box
# ─────────────────────────────────────────────────────────────────────────────
st.title("📋 Job Detail Listing Assistant")
st.caption(f"🔌 Model loaded in **{model_latency:.2f} sec**")

st.markdown("### ❓ Ask a question")
query = st.text_input("", placeholder="E.g. How many completed jobs were there yesterday?")

# ─────────────────────────────────────────────────────────────────────────────
# Execute Query
# ─────────────────────────────────────────────────────────────────────────────
if query:
    total_start = time.time()
    try:
        gen_start = time.time()
        sql = nl_to_sql(query)
        gen_end = time.time()
        sqlgen_time = gen_end - gen_start

        exec_start = time.time()
        df = pd.read_sql(sql, conn)
        exec_end = time.time()
        sqlexec_time = exec_end - exec_start
        total_time = exec_end - total_start

        # ── Show Result First (like mainui-hskp)
        if df.empty:
            st.warning("No results found.")
        else:
            if df.shape == (1, 1) and df.dtypes[0] in ['int64', 'float64']:
                val = df.iloc[0, 0]
                st.metric(label="🔢 Result", value=f"{val}")
            elif df.shape[1] == 2 and df.dtypes[1] in ['int64', 'float64']:
                st.markdown("### 📊 Chart (if applicable)")
                chart_type = st.radio("Chart type:", ["Bar Chart", "Line Chart"], horizontal=True, key="chart_type")
                df_vis = df.dropna()
                df_vis.columns = ["Label", "Value"]
                if chart_type == "Bar Chart":
                    st.bar_chart(df_vis.set_index("Label"))
                else:
                    st.line_chart(df_vis.set_index("Label"))
                with st.expander("📄 View Data Table"):
                    st.dataframe(df, use_container_width=True)
            else:
                with st.expander("📄 View Data Table"):
                    st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Result", data=csv, file_name="query_result.csv", mime="text/csv")

        # ── SQL Output (below result, like mainui)
        st.markdown("### 💬 SQL Query")
        st.code(sql, language="sql")

        st.markdown(f"<div style='font-size:1.1rem; color: #6c757d;'>⏱️ Total: <b>{total_time:.4f}s</b> | 🧠 SQL Gen: <b>{sqlgen_time:.4f}s</b> | 📦 Exec: <b>{sqlexec_time:.4f}s</b></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Toggle Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if "show_kpis" not in st.session_state:
    st.session_state.show_kpis = False

st.markdown("## 📊 KPI Dashboard")
if st.button("📊 Show/Hide KPI Section", use_container_width=True):
    st.session_state.show_kpis = not st.session_state.show_kpis

if st.session_state.show_kpis:
    st.markdown("### 🧮 KPI Overview")
    kpi_queries = {
        "📌 Total Jobs": "SELECT COUNT(*) FROM job_detail_listing;",
        "✅ Completed Jobs": "SELECT COUNT(*) FROM job_detail_listing WHERE job_status = 'Completed';",
        "🚧 Pending Jobs": "SELECT COUNT(*) FROM job_detail_listing WHERE job_status = 'Pending';",
        "⏱️ Timed-Out Jobs": "SELECT COUNT(*) FROM job_detail_listing WHERE job_status = 'Timed-Out';",
        "🏷️ Top Service Category": '''
            SELECT service_item_category FROM job_detail_listing
            GROUP BY service_item_category
            ORDER BY COUNT(*) DESC LIMIT 1;
        '''
    }
    col1, col2 = st.columns(2)
    for i, (label, query) in enumerate(kpi_queries.items()):
        with (col1 if i % 2 == 0 else col2):
            try:
                value = pd.read_sql(query, conn).iloc[0, 0]
                st.metric(label=label, value=value)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Trend Chart
    st.markdown("### 📈 Jobs in the Last 7 Days")
    try:
        df_trend = pd.read_sql("""
            SELECT date(substr("date_time_created", 1, 10)) AS day, COUNT(*) AS jobs
            FROM job_detail_listing
            WHERE "date_time_created" IS NOT NULL
              AND date(substr("date_time_created", 1, 10)) >= date('now', '-7 days')
            GROUP BY day ORDER BY day;
        """, conn)
        df_trend["day"] = pd.to_datetime(df_trend["day"], errors="coerce")
        df_trend = df_trend.dropna()
        df_trend.set_index("day", inplace=True)
        if not df_trend.empty:
            st.line_chart(df_trend["jobs"])
        else:
            st.info("No trend data available.")
    except Exception as e:
        st.error(f"Trend Error: {e}")


# what is the average credit cost per property