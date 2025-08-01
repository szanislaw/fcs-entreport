import streamlit as st
import sqlite3
import pandas as pd
import torch
import os
import time
import difflib
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─── Load model & tokenizer once ───
@st.cache_resource
def load_model_with_timing():
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")

    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    end_time = time.time()
    latency = end_time - start_time
    return tokenizer, model, latency

tokenizer, model, model_load_latency = load_model_with_timing()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model loaded on {device}")

# ─── Initialize SQLite DB ───
def initialize_database(conn, sql_file_path="schemas/cleaning.sql"):
    with open(sql_file_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()

db_path = "schemas/cleaning.db"
conn = sqlite3.connect(db_path)

required_tables = ["regions", "hotels", "rooms", "guests", "bookings", "payments", "staff",
                   "hotel_staff", "services", "room_services", "reviews",
                   "shifts", "room_cleaning", "room_inspections", "lost_found",
                   "guest_complaints", "training", "roster"]

existing_tables = set(row[0] for row in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table';").fetchall())

if not any(tbl in existing_tables for tbl in required_tables):
    initialize_database(conn, "schemas/cleaning.sql")

# ─── PostgreSQL to SQLite cleaner ───
def pg_to_sqlite(sql: str) -> str:
    cleaned = sql
    cleaned = re.sub(r'\bpublic\.', '', cleaned)
    cleaned = re.sub(r'\bTRUE\b', '1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bFALSE\b', '0', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'::\s*\w+', '', cleaned)
    cleaned = re.sub(
        r"(?i)(\b\w+\b\.)?(\b\w+\b)\s+ILIKE\s+'(.*?)'",
        lambda m: f"LOWER({m.group(1) or ''}{m.group(2)}) LIKE '%{m.group(3).lower()}%'",
        cleaned
    )
    cleaned = re.sub(r'timestamp(?:\(\d+\))?\s+without time zone', 'datetime', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bSERIAL\b', 'INTEGER', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'USING\s+btree', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r'\bAVG\s*\(\s*([a-zA-Z_][\w\.]*)\s*-\s*([a-zA-Z_][\w\.]*)\s*\)',
        r'ROUND(AVG((julianday(\1) - julianday(\2)) * 24 * 60), 2)',
        cleaned
    )
    cleaned = re.sub(
        r"date_trunc\(\s*'month'\s*,\s*([a-zA-Z_][\w\.]*)\)",
        r"strftime('%Y-%m-01', \1)",
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r'\bRETURNING\b.*?(;|\n|$)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip().rstrip(';')
    return cleaned

# ─── Generate schema dynamically ───
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
Generate a SQL query to answer the following question according to the provided schema:
{question}

Do not use any window functions. Use only the columns and tables provided in the schema. 

When generating the SQL query, ensure that:
- The query is valid SQL syntax.
- The query is optimized for SQLite.
- The query does not contain any unnecessary complexity.
- The query is directly executable in SQLite without modification.

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

### Database Schema
{schema_text}

### SQL
"""
    print(schema_text)  # Debugging: print the schema to console
    print(prompt)  # Debugging: print the prompt to console
    return prompt

def clean_sql(raw_sql: str) -> str:
    for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
        idx = raw_sql.upper().find(kw)
        if idx != -1:
            return raw_sql[idx:].strip()
    return raw_sql.strip()

def postprocess_sql_for_dual_count(sql: str, question: str) -> str:
    if (
        "COUNT" in sql.upper()
        and "room_cleaning" in sql
        and "room_inspections" in sql
        and "passed" in sql
        and re.search(r'LOWER\s*\(\s*s\.staff_name\s*\)\s+LIKE', sql, re.IGNORECASE)
    ):
        match_where = re.search(r'WHERE\s+(.+)', sql, re.IGNORECASE)
        if match_where:
            where_clause = match_where.group(1)
            new_sql = f"""
                SELECT
                    COUNT(DISTINCT rc.room_id) AS total_rooms_cleaned,
                    COUNT(DISTINCT CASE WHEN ri.passed = 1 THEN rc.room_id END) AS rooms_passed_inspection
                FROM room_cleaning rc
                JOIN staff s ON rc.staff_id = s.staff_id
                LEFT JOIN room_inspections ri ON rc.cleaning_id = ri.cleaning_id
                WHERE {where_clause.replace('AND ri.passed = 1', '')}
            """
            return new_sql.strip()
    return sql

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
    sql = postprocess_sql_for_dual_count(sql, question)
    sqlite_sql = pg_to_sqlite(sql)
    return sqlite_sql

# ─── Logo and Title ───
logo_path = "assets/fcslogo.svg"  # or .png if you prefer

logo_col, title_col = st.columns([2, 10])
with logo_col:
    st.markdown("<div style='padding-top: 30px'></div>", unsafe_allow_html=True)
    st.image(logo_path, width=120)

with title_col:
    st.title("FCS Enterprise Report Demo")
    st.caption(f"Model loaded in **{model_load_latency:.2f} seconds**")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
    }
    label[data-baseweb="input"] > div {
        font-size: 1.1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.15rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# st.title(":hotel: FCS Enterprise Report Demo")
# st.markdown(f"<div style='font-size:1.2rem; color: #6c757d;'>Model loaded in <b>{model_load_latency:.2f} seconds</b></div>", unsafe_allow_html=True)

st.markdown("### :mag_right: What can I do for you today?")
user_query = st.text_input("", placeholder="E.g. How many rooms have been cleaned?")

# ─── NL → SQL Mapping ───
def query_mapping(user_input: str) -> str | None:
    mappings = {
        "average number of rooms cleaned per shift": """
            SELECT ROUND(AVG(rooms_cleaned), 2) AS avg_rooms_cleaned FROM shifts;
        """,
        "average cleaning time per room": """
            SELECT ROUND(AVG((julianday(cleaning_end) - julianday(cleaning_start)) * 24 * 60), 2) AS avg_cleaning_time_min FROM room_cleaning;
        """,
        "room inspection pass rate": """
            SELECT ROUND(SUM(CASE WHEN passed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pass_rate FROM room_inspections;
        """,
        "how many rooms did donald anderson clean and pass the inspection": """
            SELECT
                COUNT(DISTINCT rc.room_id) AS total_rooms_cleaned,
                COUNT(DISTINCT CASE WHEN ri.passed = 1 THEN rc.room_id END) AS rooms_passed_inspection
            FROM room_cleaning rc
            JOIN staff s ON rc.staff_id = s.staff_id
            LEFT JOIN room_inspections ri ON rc.cleaning_id = ri.cleaning_id
            WHERE LOWER(s.staff_name) LIKE '%donald anderson%'
        """
    }

    # Lowercase keys and input
    user_input_clean = user_input.lower().strip()
    phrases = list(mappings.keys())

    # Fuzzy match threshold
    best_match = difflib.get_close_matches(user_input_clean, phrases, n=1, cutoff=0.7)
    if best_match:
        return mappings[best_match[0]].strip()

    return None



if user_query:
    try:
        sql = query_mapping(user_query)
        if not sql:
            sql = nl_to_sql(user_query)

        if not sql.strip():
            raise ValueError("The generated SQL query is empty.")

        try:
            start_time = time.time()
            df = pd.read_sql(sql, conn)
            end_time = time.time()
            latency = end_time - start_time

            st.success("Query successful.")

            # Begin result display
            if not df.empty:
                st.markdown("#### 📊 Result")

                # Check if it's a room cleaning + inspection summary
                cols = df.columns.str.lower().tolist()
                if set(['total_rooms_cleaned', 'rooms_passed_inspection']).issubset(cols):
                    cleaned = df.iloc[0][cols.index('total_rooms_cleaned')]
                    passed = df.iloc[0][cols.index('rooms_passed_inspection')]
                    st.metric(label="🧹 Room Cleaning Summary", value=f"{int(cleaned)} rooms cleaned, {int(passed)} rooms passed")

                elif df.shape[0] == 1 and all(dtype in ['int64', 'float64'] for dtype in df.dtypes):
                    for col in df.columns:
                        val = df[col].iloc[0]
                        label = col.replace('_', ' ').title()

                        # Format as percentage if column name suggests it's a rate
                        if "rate" in col.lower() or "percentage" in col.lower():
                            display_val = f"{val * 100:.2f}%" if isinstance(val, (float, int)) else val
                        else:
                            display_val = f"{val:.2f}" if isinstance(val, float) else val

                        st.metric(label=label, value=display_val)

                else:
                    st.success(f"✅ Returned {df.shape[0]} rows and {df.shape[1]} columns.")

                    # Bar chart if suitable
                    if df.shape[0] <= 10 and df.shape[1] == 2 and df.dtypes[1] in ['int64', 'float64']:
                        st.markdown("#### 📊 Bar Chart")
                        st.bar_chart(df.set_index(df.columns[0]))

                    # Summary statistics
                    if any(dtype in ['int64', 'float64'] for dtype in df.dtypes):
                        st.markdown("#### 📈 Summary Statistics")
                        st.dataframe(df.describe())

                    # Expandable full table
                    with st.expander("🔍 View Full Data Table"):
                        st.dataframe(df, use_container_width=True)

                # CSV download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Result as CSV", data=csv, file_name="query_result.csv", mime="text/csv")

                # 🔽 Moved to bottom
                st.markdown("#### 🧠 Generated SQL Query")
                st.code(sql, language="sql")

            else:
                st.warning("No results found.")

            # Show latency at bottom
            st.markdown(f"<div style='font-size:1.2rem; color: #6c757d;'>⏱️ Query time: <b>{latency:.4f} seconds</b></div>", unsafe_allow_html=True)

            
        except Exception as sql_error:
            st.error("❌ Something went wrong while executing your query.")
            with st.expander("Show full error"):
                st.code(str(sql_error), language="text")

    except Exception as gen_error:
        st.error(f"❌ Failed to generate a valid SQL query:\n{gen_error}")

# ─── Housekeeping KPI Dashboard ──────────────────────────────────────────────

# Optional CSS to help center the button more accurately
st.markdown("""
    <style>
    div[data-testid="column"] div.stButton {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Initialize toggle state ───
if "show_kpis" not in st.session_state:
    st.session_state.show_kpis = False

# ─── Centered Toggle Button ───
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("📊 Show/Hide Housekeeping KPIs", use_container_width=True):
        st.session_state.show_kpis = not st.session_state.show_kpis

# ─── Conditional Display ───
if st.session_state.show_kpis:
    st.markdown("## A brief look at your Housekeeping KPIs...")

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
