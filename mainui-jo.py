import re
import streamlit as st
import sqlite3
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

# ─────────────────────────────────────────────────────────────────────────────
# Optional: Build .db from .sql file
# ─────────────────────────────────────────────────────────────────────────────
def build_db_from_sql(sql_file: str, db_file: str):
    if not os.path.exists(db_file) and os.path.exists(sql_file):
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        conn = sqlite3.connect(db_file)
        try:
            conn.executescript(sql_script)
            print(f"✅ Built database '{db_file}' from '{sql_file}'")
        except Exception as e:
            print(f"❌ Failed to build DB: {e}")
        finally:
            conn.close()

# Path settings
SQL_PATH = "job_detail_listing.sql"
DB_PATH = "job_detail_listing.db"

# Build database from .sql file if needed
build_db_from_sql(SQL_PATH, DB_PATH)

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
conn = sqlite3.connect("job_detail_listing.db")

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
    prompt = f"""
### Task
Generate a SQL query to answer the following question according to the provided schema below strictly:
{question}

### Database Schema
{schema_text}

Do not use any window functions. Use only the columns and tables provided in the schema. 

When generating the SQL query, ensure that:
- The query is valid SQL syntax.
- The query is optimized for SQLite.
- The query does not contain any unnecessary complexity.

If the verbose of the question is not clear, return an empty result set.

If the question is not answerable with the provided schema, return an empty result set. 
If the question is not clear, return an empty result set. 
If the query is not a question, return an empty result set. 
If there are multiple tables, you may need to join them. 
If the question is about a specific table, use only that table.
If the question is about a specific column, use only that column.
If the question is about a specific value, use only that value.
If the question is about a specific date, use only that date.
If the question is about a specific time range, use only that time range.
If the question is about a specific person, use only that person.
If the question is about a specific location, use only that location.
If the question is about a specific service, use only that service.

For context, the database is a job detail listing system. The tables contain information about jobs, their statuses, service categories, and timestamps.

Jobs could be referred to as "service items", "service requests" or "calls". The job statuses include "Completed", "Pending", and "Timed-Out". 
Do not assume jobs are only completed; they can also be pending or timed out.
The service categories include various types of services provided. 
Try not to use date time functions unless necessary, as the database is SQLite and may not support all date functions.
Always use the date_time_created_ column for date-related queries. 
When asked about averages, use the date_time_created_ column to analyze dates and times of job openings and completions. 
When asked 'What', assume the question is about the job detail listing system and its data.  
When asked about trends, try to analyze the job completion trends over time, such as daily or monthly trends.
Try to output a 2 column table with the first column as the date and the second column as the number of jobs completed on that date where possible.
When asked 'per room', always use the location column to filter jobs by room and output in a 2 column table with the first column as the location and the second column as the number of jobs completed in that room.
Location can be a room, a building, or a specific area within the job detail listing system.
Service items should refer to the service_item_category column in the job detail listing system.


### SQL
"""
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

    import re

    # Fix ILIKE → LOWER(...) LIKE
    sql = re.sub(
        r"(\w+(?:\.\w+)?)\s+ILIKE\s+'(.*?)'",
        lambda m: f"LOWER({m.group(1)}) LIKE '%{m.group(2).lower()}%'",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(YEAR FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(MONTH FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%m', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(DAY FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*DAY\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%d', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(DOW FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*DOW\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%w', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(HOUR FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*HOUR\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%H', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(MINUTE FROM col)
    sql = re.sub(
        r"EXTRACT\s*\(\s*MINUTE\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%M', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix PostgreSQL-style casting (col::TYPE) → CAST(col AS TYPE)
    sql = re.sub(
        r"(\w+(?:\.\w+)?)::(\w+)",
        r"CAST(\1 AS \2)",
        sql
    )

    # Fix date_trunc('day', to_timestamp(col)) → date(substr(col, 1, 10))
    sql = re.sub(
        r"date_trunc\s*\(\s*'day'\s*,\s*to_timestamp\s*\(([^)]+)\)\s*\)",
        r"date(substr(\1, 1, 10))",
        sql,
        flags=re.IGNORECASE
    )

    # Fix strftime('%X', to_timestamp(col)) → strftime('%X', col)
    sql = re.sub(
        r"strftime\(\s*'(%[YmdwHMS])'\s*,\s*to_timestamp\(([^)]+)\)\s*\)",
        r"strftime('\1', \2)",
        sql,
        flags=re.IGNORECASE
    )

    # Final fallback: remove remaining to_timestamp(col) → col
    sql = re.sub(
        r"to_timestamp\s*\(([^)]+)\)",
        r"\1",
        sql,
        flags=re.IGNORECASE
    )

    # Fix generic date_trunc('day', col) → date(substr(col, 1, 10))
    sql = re.sub(
        r"date_trunc\s*\(\s*'day'\s*,\s*([^)]+?)\s*\)",
        r"date(substr(\1, 1, 10))",
        sql,
        flags=re.IGNORECASE
    )

    # Fix date_trunc('month', col)
    sql = re.sub(
        r"date_trunc\s*\(\s*'month'\s*,\s*([^)]+?)\s*\)",
        r"date(substr(\1, 1, 7) || '-01')",
        sql,
        flags=re.IGNORECASE
    )

    # Fix date_part('year', col)
    sql = re.sub(
        r"date_part\s*\(\s*'year'\s*,\s*([^)]+?)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE
    )

    # Fix INTERVAL 'N unit' → date('now', '-N unit') or datetime(...)
    sql = re.sub(
        r"(CURRENT_DATE|CURRENT_TIMESTAMP)\s*-\s*INTERVAL\s*'(\d+)\s+(day|days|week|weeks|month|months|year|years)'",
        lambda m: f"{'date' if m.group(1).upper() == 'CURRENT_DATE' else 'datetime'}('now', '-{m.group(2)} {m.group(3)}')",
        sql,
        flags=re.IGNORECASE
    )

    # Fix datetime subtraction: col1 - col2 → julianday(col1) - julianday(col2)
    sql = re.sub(
        r"\b([a-zA-Z_][\w\.]*)\s*-\s*([a-zA-Z_][\w\.]*)\b",
        r"julianday(\1) - julianday(\2)",
        sql
    )

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