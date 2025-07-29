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

    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        torch_dtype=torch.float16,  # Use float16 for speed; omit for float32
        device_map="auto"
    )

    end_time = time.time()
    latency = end_time - start_time
    return tokenizer, model, latency

tokenizer, model, model_load_latency = load_model_with_timing()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded on {device}")

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

# Only initialize if none of the required tables exist
if not any(tbl in existing_tables for tbl in required_tables):
    initialize_database(conn, "hotel.sql")


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

