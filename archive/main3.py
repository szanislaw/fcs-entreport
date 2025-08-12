import streamlit as st
import sqlite3
import pandas as pd
import torch
import os
import time
import re

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
print(f"Model loaded on {device} in {model_load_latency:.2f} seconds")

# ─── Initialize SQLite DB ────────────────────────────────────────────────────
def initialize_database(conn, sql_file_path="hotel.sql"):
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()

db_path = "hotel.db"
conn = sqlite3.connect(db_path)

required_tables = ["regions", "hotels", "rooms", "guests", "bookings", "payments", "staff",
                   "hotel_staff", "services", "room_services", "reviews",
                   "shifts", "room_cleaning", "room_inspections", "lost_found",
                   "guest_complaints", "training", "roster"]

existing_tables = set(row[0] for row in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table';").fetchall())

if not any(tbl in existing_tables for tbl in required_tables):
    initialize_database(conn, "hotel.sql")


# ─── PostgreSQL to SQLite cleaner ───────────────────────────────────────────
import re

def pg_to_sqlite(sql: str) -> str:
    """
    Convert PostgreSQL-style SQL into SQLite-compatible SQL.
    Handles common syntax and function mismatches.
    """
    cleaned = sql

    # ─── Schema prefixes (e.g., public.table) ────────────────────────────────
    cleaned = re.sub(r'\bpublic\.', '', cleaned)

    # ─── Boolean literals ────────────────────────────────────────────────────
    cleaned = re.sub(r'\bTRUE\b', '1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bFALSE\b', '0', cleaned, flags=re.IGNORECASE)

    # ─── Type casts (e.g., column::type) ─────────────────────────────────────
    cleaned = re.sub(r'::\s*\w+', '', cleaned)

    # ─── Replace timestamp type declarations ─────────────────────────────────
    cleaned = re.sub(r'timestamp(?:\(\d+\))?\s+without time zone', 'datetime', cleaned, flags=re.IGNORECASE)

    # ─── SERIAL → INTEGER (note: SQLite uses INTEGER PRIMARY KEY AUTOINCREMENT for actual auto-increment) ───
    cleaned = re.sub(r'\bSERIAL\b', 'INTEGER', cleaned, flags=re.IGNORECASE)

    # ─── USING btree (irrelevant in SQLite) ──────────────────────────────────
    cleaned = re.sub(r'USING\s+btree', '', cleaned, flags=re.IGNORECASE)

    # ─── Replace PostgreSQL-style interval math: AVG(end - start) → julianday math ──────
    cleaned = re.sub(
        r'\bAVG\s*\(\s*([a-zA-Z_][\w\.]*)\s*-\s*([a-zA-Z_][\w\.]*)\s*\)',
        r'ROUND(AVG((julianday(\1) - julianday(\2)) * 24 * 60), 2)',
        cleaned
    )

    # ─── Replace date_trunc (not in SQLite) with strftime if pattern matches ───────────
    cleaned = re.sub(
        r"date_trunc\(\s*'month'\s*,\s*([a-zA-Z_][\w\.]*)\)",
        r"strftime('%Y-%m-01', \1)",
        cleaned,
        flags=re.IGNORECASE
    )

    # ─── Optional: remove unsupported RETURNING clause ───────────────────────
    cleaned = re.sub(r'\bRETURNING\b.*?(;|\n|$)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # ─── Strip trailing semicolon (sqlite3 doesn't require it in query string) ────────
    cleaned = cleaned.strip().rstrip(';')

    return cleaned

# ─── Generate schema dynamically ─────────────────────────────────────────────
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
    sql = clean_sql(raw)
    sqlite_sql = pg_to_sqlite(sql)
    return sqlite_sql

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
            if not df.empty:
                # If single-row numeric output
                if df.shape[0] == 1 and all(dtype in ['int64', 'float64'] for dtype in df.dtypes):
                    for col in df.columns:
                        val = df[col].iloc[0]
                        st.metric(label=col.replace('_', ' ').title(), value=f"{val:.2f}" if isinstance(val, float) else val)

                    # Pie chart if it's a rate column (percentage)
                    if any("rate" in col.lower() or "percentage" in col.lower() for col in df.columns):
                        rate_col = df.columns[0]
                        rate_val = float(df[rate_col].iloc[0])
                        st.markdown("#### 📊 Breakdown")
                        st.pyplot(generate_pie_chart(rate_val, label=rate_col))
                else:
                    st.success(f"✅ Returned {df.shape[0]} rows and {df.shape[1]} columns.")

                    # Show bar chart if it looks like a ranking table
                    if df.shape[0] <= 10 and df.shape[1] == 2 and df.dtypes[1] in ['int64', 'float64']:
                        st.markdown("#### 📊 Bar Chart")
                        st.bar_chart(df.set_index(df.columns[0]))

                    # Show summary stats for numeric columns
                    if any(dtype in ['int64', 'float64'] for dtype in df.dtypes):
                        st.markdown("#### 📈 Summary Statistics")
                        st.dataframe(df.describe())

                    # Expandable full table
                    with st.expander("🔍 View Full Data Table"):
                        st.dataframe(df, use_container_width=True)

                # Download option
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Result as CSV", data=csv, file_name="query_result.csv", mime="text/csv")
            else:
                st.warning("No results found.")

            st.caption(f"⏱️ Query time: **{latency:.4f} seconds**")
        except Exception as sql_error:
            st.error("❌ Something went wrong while executing your query.")
            with st.expander("Show full error"):
                st.code(str(sql_error), language="text")

    except Exception as gen_error:
        st.error(f"❌ Failed to generate a valid SQL query:\n{gen_error}")

# ─── Housekeeping KPI Dashboard ──────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🧹 Housekeeping KPI Dashboard")

kpi_queries = {
    "🧺 Avg. Rooms Cleaned per Shift": """
        SELECT ROUND(AVG(rooms_cleaned), 2) AS avg_rooms_cleaned
        FROM shifts;
    """,
    "⏱️ Avg. Cleaning Time per Room (min)": """
        SELECT ROUND(AVG((julianday(cleaning_end) - julianday(cleaning_start)) * 24 * 60), 2) AS avg_cleaning_time_min
        FROM room_cleaning;
    """,
    "✅ Room Inspection Pass Rate (%)": """
        SELECT ROUND(SUM(CASE WHEN passed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pass_rate
        FROM room_inspections;
    """,
    "🔁 Room Re-Clean Rate (%)": """
        SELECT ROUND(SUM(CASE WHEN re_clean_required THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS re_clean_rate
        FROM room_cleaning;
    """,
    "🚨 Turnaround Time for Rush Rooms (min)": """
        SELECT ROUND(AVG((julianday(cleaning_end) - julianday(cleaning_start)) * 24 * 60), 2) AS avg_rush_time
        FROM room_cleaning
        WHERE is_rush = 1;
    """,
    "📦 Lost & Found Resolution Time (hrs)": """
        SELECT ROUND(AVG((julianday(resolution_time) - julianday(report_time)) * 24), 2) AS avg_resolution_time_hrs
        FROM lost_found;
    """,
    "😠 Guest Complaints Count": """
        SELECT COUNT(*) AS total_complaints
        FROM guest_complaints;
    """,
    "📈 Attendant Productivity (Rooms/Shift)": """
        SELECT ROUND(AVG(rooms_cleaned), 2) AS productivity
        FROM shifts;
    """,
    "🕒 Total Overtime Hours (Monthly)": """
        SELECT ROUND(SUM(overtime_hours), 2) AS total_overtime_hours
        FROM shifts
        WHERE strftime('%Y-%m', shift_date) = '2025-07';
    """,
    "🤒 Sick Leave Rate (%)": """
        SELECT ROUND(SUM(CASE WHEN sick_leave THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS sick_rate
        FROM shifts;
    """,
    "📊 Staffing Coverage vs Roster (%)": """
        SELECT ROUND(SUM(actual_hours) * 100.0 / SUM(scheduled_hours), 2) AS coverage_rate
        FROM roster;
    """,
    "🎓 Avg. Training Hours per Staff": """
        SELECT ROUND(AVG(hours), 2) AS avg_training_hours
        FROM training;
    """
}

col1, col2 = st.columns(2)

for idx, (title, query) in enumerate(kpi_queries.items()):
    with (col1 if idx % 2 == 0 else col2):
        st.markdown(f"### {title}")
        try:
            start = time.time()
            df_kpi = pd.read_sql(query, conn)
            end = time.time()
            if not df_kpi.empty:
                value = df_kpi.iloc[0, 0]
                formatted_value = f"{value:.2f}" if isinstance(value, (float, int)) else str(value)
                st.metric(label=title, value=formatted_value)
                st.caption(f"⏱️ Loaded in {end - start:.4f} sec")
            else:
                st.warning("No data available.")
        except Exception as e:
            st.error(f"Error loading KPI: {e}")

# ─── Top 3 Highlights ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🏆 Top 3 Highlights")

top_kpi_queries = {
    "🥇 Top 3 Staff by Rooms Cleaned": """
        SELECT s.staff_name, SUM(sh.rooms_cleaned) AS total_cleaned
        FROM shifts sh
        JOIN staff s ON sh.staff_id = s.staff_id
        GROUP BY sh.staff_id
        ORDER BY total_cleaned DESC
        LIMIT 3;
    """,
    "🧽 Top 3 Rooms with Most Re-cleans": """
        SELECT rc.room_id, COUNT(*) AS re_cleans
        FROM room_cleaning rc
        WHERE rc.re_clean_required = 1
        GROUP BY rc.room_id
        ORDER BY re_cleans DESC
        LIMIT 3;
    """,
    "😠 Rooms with Most Complaints": """
        SELECT gc.room_id, COUNT(*) AS complaints
        FROM guest_complaints gc
        GROUP BY gc.room_id
        ORDER BY complaints DESC
        LIMIT 3;
    """
}

for title, query in top_kpi_queries.items():
    st.markdown(f"### {title}")
    try:
        df = pd.read_sql(query, conn)
        if not df.empty:
            for idx, row in df.iterrows():
                rank = idx + 1
                label = f"{rank}. {row[0]}"
                value = row[1]
                st.metric(label=label, value=value)
        else:
            st.info("No data available.")
    except Exception as e:
        st.error(f"Failed to load {title}: {e}")
        
        
        
# show me a chart comparing number of rooms cleaned by staff
