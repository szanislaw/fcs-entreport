import streamlit as st
import sqlite3
import pandas as pd
import torch
import os
import time
import difflib
import re
import plotly.express as px
import altair as alt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Limit centered content to around 90% or a custom max width */
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
        max-width: 90rem;       /* adjust as desired (e.g., 80rem, 75rem, etc.) */
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Expand the main container */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

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
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,                                  #can remove this line if pad_token_id is not needed
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
            total_start_time = time.time()

            # Time the SQL generation
            sqlgen_start_time = time.time()
            sql = query_mapping(user_query)
            if not sql:
                sql = nl_to_sql(user_query)
            sqlgen_end_time = time.time()
            sqlgen_latency = sqlgen_end_time - sqlgen_start_time

            # Time the SQL execution
            sqlexec_start_time = time.time()
            df = pd.read_sql(sql, conn)
            sqlexec_end_time = time.time()
            sqlexec_latency = sqlexec_end_time - sqlexec_start_time

            total_latency = sqlexec_end_time - total_start_time


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
                    
                    
                    ##not sure if this is needed, but keeping it for now
                    if df.shape[0] <= 10 and df.shape[1] == 2 and df.dtypes[1] in ['int64', 'float64']:
                        st.markdown("#### 📊 Bar Chart")
                        st.bar_chart(df.set_index(df.columns[0]))


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
            st.markdown(f"<div style='font-size:1.2rem; color: #6c757d;'>⏱️ Total time: <b>{total_latency:.4f} seconds</b> &nbsp; | &nbsp; 🧠 SQL generation: <b>{sqlgen_latency:.4f} sec</b> &nbsp; | &nbsp; 📦 SQL execution: <b>{sqlexec_latency:.4f} sec</b></div>", unsafe_allow_html=True)

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
    # Track KPI timing
    kpi_total_start_time = time.time()
    kpi_sqlgen_start_time = kpi_total_start_time
    kpi_sqlgen_end_time = kpi_total_start_time  # assuming no SQL generation phase here

    col1, col2 = st.columns([1, 1], gap="large")  # Wider and spaced columns

    with col1:
        st.markdown("#### 👤 Top 5 Staff by Number of Assignments")
        query_staff = """
            SELECT assigned_name, COUNT(*) AS assignments
            FROM hskp_cleaning_order
            WHERE assigned_name IS NOT NULL AND TRIM(assigned_name) <> ''
            GROUP BY assigned_name
            ORDER BY assignments DESC
            LIMIT 5;
        """
        df_staff = pd.read_sql(query_staff, conn)

        if not df_staff.empty and "assigned_name" in df_staff.columns:
            fig = px.bar(
                df_staff,
                x="assigned_name",
                y="assignments",
                text="assignments",
                labels={"assigned_name": "Staff Member", "assignments": "No. of Assignments"},
                title="Top 5 Staff by Cleaning Assignments",
            )
            fig.update_traces(marker_color="#1f77b4", textposition="outside")
            fig.update_layout(
                xaxis_tickfont=dict(size=14),
                yaxis_title="Number of Assignments",
                xaxis_title="Staff Name",
                title_font_size=18,
                height=450,
                margin=dict(l=30, r=30, t=50, b=100),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No staff assignment data available.")

    with col2:
        st.markdown("#### 🧼 Top 5 Cleaning Services by Type")
        query_service_type = """
            SELECT service_type, COUNT(*) AS total
            FROM hskp_cleaning_order
            WHERE service_type IS NOT NULL AND TRIM(service_type) <> ''
            GROUP BY service_type
            ORDER BY total DESC
            LIMIT 5;
        """
        df_service_type = pd.read_sql(query_service_type, conn)

        if not df_service_type.empty:
            chart_services = alt.Chart(df_service_type).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X("service_type:N", sort='-y', title="Service Type"),
                y=alt.Y("total:Q", title="Number of Tasks"),
                color=alt.value("#4e79a7"),
                tooltip=["service_type", "total"]
            ).properties(
                height=350,
                title="Top 5 Cleaning Services by Type"
            )
            st.altair_chart(chart_services, use_container_width=True)
        else:
            st.warning("No service type data available.")
    

    # ─── Row 2: Full-width Time vs Credit ───
    st.markdown("#### ⏱️ Average Time to Completion vs. Credit")
    query_time_vs_credit = """
        SELECT credit, AVG(time_spent) AS avg_time_spent
        FROM hskp_cleaning_order
        WHERE credit IS NOT NULL AND time_spent IS NOT NULL
        GROUP BY credit
        ORDER BY credit ASC;
    """
    df_time_vs_credit = pd.read_sql(query_time_vs_credit, conn)

    if not df_time_vs_credit.empty:
        chart_time_credit = alt.Chart(df_time_vs_credit).mark_line(point=True).encode(
            x=alt.X("credit:Q", title="Credit"),
            y=alt.Y("avg_time_spent:Q", title="Avg. Time Spent (mins)"),
            tooltip=["credit", "avg_time_spent"]
        ).properties(
            height=350,
            title="Average Time to Completion vs. Credit"
        )
        st.altair_chart(chart_time_credit, use_container_width=True)
    else:
        st.warning("No time vs credit data available.")





    # KPI: Daily Cleaning Volume (Past 7 Days) - demonstration of failback handling
    st.markdown("#### 🧹 Daily Cleaning Volume (Past 7 Days)")

    query = """
        SELECT DATE(created_date) AS day, COUNT(*) AS total
        FROM hskp_cleaning_order
        WHERE created_date IS NOT NULL AND TRIM(created_date) <> ''
        AND DATE(created_date) >= DATE('now', '-7 days')
        GROUP BY day
        ORDER BY day;
    """

    df_daily = pd.read_sql(query, conn)

    if not df_daily.empty and "day" in df_daily.columns:
        df_daily["day"] = pd.to_datetime(df_daily["day"], errors='coerce')
        df_daily = df_daily.dropna(subset=["day"])
        df_daily.set_index("day", inplace=True)
        st.line_chart(df_daily["total"])
    else:
        st.warning("No daily cleaning data available for the past 7 days.")

    
    st.markdown("## 🧠 Top 10 Strategic Housekeeping KPIs")

    kpi_queries = {
        "🧼 Cleaning Order Completion Rate (%)": """
            SELECT ROUND(
                SUM(CASE WHEN job_stop IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) AS completion_rate
            FROM hskp_cleaning_order;
        """,

        "🧑‍💼 Avg. Cleanings per Attendant per Day": """
            SELECT ROUND(AVG(cnt), 2) FROM (
                SELECT COUNT(*) AS cnt
                FROM hskp_cleaning_order_detail
                GROUP BY user_uuid, DATE(created_date)
            );
        """,

        "📉 % of Cleanings Cancelled": """
            SELECT ROUND(
                SUM(CASE WHEN cancelled_date IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) AS cancellation_rate
            FROM hskp_cleaning_order;
        """,

        "👤 Most Frequently Assigned Staff": """
            SELECT assigned_name, COUNT(*) AS assignments
            FROM hskp_cleaning_order
            WHERE assigned_name IS NOT NULL
            GROUP BY assigned_name
            ORDER BY assignments DESC
            LIMIT 1;
        """,

        "🚩 % of Tasks with Special Remarks": """
            SELECT ROUND(
                SUM(CASE WHEN remarks IS NOT NULL AND TRIM(remarks) <> '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) AS flagged_percentage
            FROM hskp_cleaning_order;
        """,

        "🏷️ Most Common Additional Task Type": """
            SELECT additional_task_id, COUNT(*) AS freq
            FROM hskp_cleaning_order_map_additional_task
            GROUP BY additional_task_id
            ORDER BY freq DESC
            LIMIT 1;
        """,

        "📋 Avg. Checklist Score (Where Available)": """
            SELECT ROUND(AVG(score), 2) AS avg_score
            FROM hskp_cleaning_order_map_checklist
            WHERE score IS NOT NULL;
        """,

        "📉 Inspection Failure Rate (%)": """
            SELECT ROUND(
                SUM(CASE WHEN inspection_result = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) AS failure_rate
            FROM hskp_cleaning_order_inspection;
        """,

        "🔁 Repeat Cleanings on Same Room in 24hrs": """
            SELECT COUNT(*) FROM (
                SELECT location_uuid, DATE(created_date) AS day, COUNT(*) AS cnt
                FROM hskp_cleaning_order
                GROUP BY location_uuid, day
                HAVING cnt > 1
            );
        """,

        "🕓 % of Cleanings Exceeding Allocated Time": """
            SELECT ROUND(
                SUM(CASE
                    WHEN job_stop IS NOT NULL AND duration IS NOT NULL AND
                         (julianday(job_stop) - julianday(job_start)) * 24 * 60 > duration
                    THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
            ) AS overrun_rate
            FROM hskp_cleaning_order
            WHERE job_start IS NOT NULL AND job_stop IS NOT NULL;
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
                

    kpi_total_end_time = time.time()
    kpi_sqlexec_latency = kpi_total_end_time - kpi_sqlgen_end_time
    kpi_total_latency = kpi_total_end_time - kpi_total_start_time

    st.markdown(
        f"<div style='font-size:1.1rem; color: #6c757d;'>⏱️ Total time: <b>{kpi_total_latency:.4f} seconds</b> &nbsp; | &nbsp; 🧠 SQL generation: <b>{0.0000:.4f} sec</b> &nbsp; | &nbsp; 📦 SQL execution: <b>{kpi_sqlexec_latency:.4f} sec</b></div>",
        unsafe_allow_html=True
    )
                    

    # KPI: Top 5 Staff by Number of Assignments
    # st.markdown("#### 👤 Top 5 Staff by Number of Assignments")

    # query_staff = """
    #     SELECT assigned_name, COUNT(*) AS assignments
    #     FROM hskp_cleaning_order
    #     WHERE assigned_name IS NOT NULL AND TRIM(assigned_name) <> ''
    #     GROUP BY assigned_name
    #     ORDER BY assignments DESC
    #     LIMIT 5;
    # """

    # df_staff = pd.read_sql(query_staff, conn)

    # if not df_staff.empty and "assigned_name" in df_staff.columns:
    #     df_staff.set_index("assigned_name", inplace=True)
    #     st.bar_chart(df_staff, use_container_width=True)
    # else:
    #     st.warning("No staff assignment data available.")
    
    
    # st.markdown("#### 👤 Top 5 Staff by Number of Assignments")

    # query_staff = """
    #     SELECT assigned_name, COUNT(*) AS assignments
    #     FROM hskp_cleaning_order
    #     WHERE assigned_name IS NOT NULL AND TRIM(assigned_name) <> ''
    #     GROUP BY assigned_name
    #     ORDER BY assignments DESC
    #     LIMIT 5;
    # """

    # df_staff = pd.read_sql(query_staff, conn)

    # if not df_staff.empty and "assigned_name" in df_staff.columns:
    #     fig = px.bar(
    #         df_staff,
    #         x="assigned_name",
    #         y="assignments",
    #         text="assignments",
    #         labels={"assigned_name": "Staff Member", "assignments": "No. of Assignments"},
    #         title="Top 5 Staff by Cleaning Assignments",
    #     )
    #     fig.update_traces(marker_color="#1f77b4", textposition="outside")
    #     fig.update_layout(
    #         xaxis_tickfont=dict(size=14),
    #         yaxis_title="Number of Assignments",
    #         xaxis_title="Staff Name",
    #         title_font_size=20,
    #         margin=dict(l=40, r=40, t=60, b=120),
    #         height=450
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.warning("No staff assignment data available.")
        
    # st.markdown("#### 🧼 Top 5 Cleaning Services by Type")

    # query_service_type = """
    #     SELECT service_type, COUNT(*) AS total
    #     FROM hskp_cleaning_order
    #     WHERE service_type IS NOT NULL AND TRIM(service_type) <> ''
    #     GROUP BY service_type
    #     ORDER BY total DESC
    #     LIMIT 5;
    # """
    # df_service_type = pd.read_sql(query_service_type, conn)

    # if not df_service_type.empty:
    #     chart_services = alt.Chart(df_service_type).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    #         x=alt.X("service_type:N", sort='-y', title="Service Type"),
    #         y=alt.Y("total:Q", title="Number of Tasks"),
    #         color=alt.value("#4e79a7"),
    #         tooltip=["service_type", "total"]
    #     ).properties(
    #         width=600,
    #         height=350,
    #         title="Top 5 Cleaning Services by Type"
    #     )
    #     st.altair_chart(chart_services, use_container_width=True)
    # else:
    #     st.warning("No service type data available.")

    # st.markdown("#### ⏱️ Average Time to Completion vs. Credit")

    # query_time_vs_credit = """
    #     SELECT credit, AVG(time_spent) AS avg_time_spent
    #     FROM hskp_cleaning_order
    #     WHERE credit IS NOT NULL AND time_spent IS NOT NULL
    #     GROUP BY credit
    #     ORDER BY credit ASC;
    # """
    # df_time_vs_credit = pd.read_sql(query_time_vs_credit, conn)

    # if not df_time_vs_credit.empty:
    #     chart_time_credit = alt.Chart(df_time_vs_credit).mark_line(point=True).encode(
    #         x=alt.X("credit:Q", title="Credit"),
    #         y=alt.Y("avg_time_spent:Q", title="Avg. Time Spent (mins)"),
    #         tooltip=["credit", "avg_time_spent"]
    #     ).properties(
    #         width=600,
    #         height=350,
    #         title="Average Time to Completion vs. Credit"
    #     )
    #     st.altair_chart(chart_time_credit, use_container_width=True)
    # else:
    #     st.warning("No time vs credit data available.")


