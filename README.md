# FCS EntReport

Enterprise Reporting using Streamlit

## Overview

FCS EntReport is an enterprise-level reporting application that leverages [Streamlit](https://streamlit.io/) for interactive dashboards and integrates AI-powered SQL generation via large language models. The application is designed to provide flexible, natural language-driven business intelligence and reporting, primarily for hotel management scenarios.

The main entry point for the application is [`mainui.py`](mainui.py).

## Key Features

- **AI-Powered SQL Generation:** Utilizes the `defog/sqlcoder-7b-2` language model (via HuggingFace Transformers) to generate SQL queries from natural language.
- **Database Agnostic:** Includes utilities to convert PostgreSQL-style SQL to SQLite-compatible queries.
- **One-Click Demo Database:** Automatically initializes a rich SQLite database (`hotel.db`) from a provided schema (`hotel.sql`) if not already present.
- **Streamlit UI:** Provides a modern, browser-based interface for interacting with data, generating reports, and visualizing results.
- **Enterprise Hotel Management Data Model:** Supports a comprehensive schema including guests, bookings, payments, staff, services, reviews, and more.

## How It Works

The workflow in `mainui.py` can be broken down into the following steps:

### 1. Model & Tokenizer Initialization

- The application loads the [defog/sqlcoder-7b-2](https://huggingface.co/defog/sqlcoder-7b-2) large language model (LLM) and tokenizer using HuggingFace's Transformers library.
- The model is loaded on GPU (CUDA) if available, otherwise CPU, utilizing float16 precision for efficiency.
- Loading is cached to avoid repeated initialization.

### 2. Database Initialization

- The app connects to a local SQLite database (`hotel.db`).
- It checks for the presence of required tables (e.g., guests, bookings, staff).
- If any are missing, it runs the provided schema in `hotel.sql` to create and populate the database.

### 3. User Query Handling & Prompt Formation

- The user enters a natural-language question via a Streamlit text input.
- The function `get_dynamic_schema_prompt(conn, question)` is called:
  - It reads the current database schema: for each table, it gets columns and types.
  - It builds a schema summary as a series of `CREATE TABLE ...` statements.
  - It then forms a prompt like:

    ```
    ### Task
    Generate a SQL query to answer the following question:
    {user_question}

    ### Database Schema
    CREATE TABLE ...;
    CREATE TABLE ...;
    ...
    ### SQL
    ```

### 4. SQL Query Generation with SQLCoder

- The prompt is tokenized and fed into the SQLCoder LLM:

  ```python
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  outputs = model.generate(
      **inputs,
      max_new_tokens=256,
      do_sample=True,
      temperature=0.7,
      top_p=0.9
  )
  raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
  ```
- The output is post-processed (`clean_sql`) to extract the SQL code from the generated text, ensuring only valid SQL is kept.
- The result is optionally further processed (e.g., custom logic for special cases like dual room cleaning/inspection counts).

### 5. SQL Dialect Conversion (Will be removed when implemented into PostgreSQL)

- The generated SQL (which may use PostgreSQL-specific syntax) is converted to SQLite-compatible SQL using `pg_to_sqlite`.
- This function handles:
  - Removing schema prefixes (e.g., `public.`)
  - Converting boolean values (`TRUE`/`FALSE` to `1`/`0`)
  - Removing typecasts
  - Rewriting `ILIKE` to `LOWER(col) LIKE ...`
  - Other necessary conversions for SQLite compatibility

### 6. Query Execution & Display

- The final SQL is executed against the SQLite database using Pandas:

  ```python
  df = pd.read_sql(sql, conn)
  ```
- If execution is successful and results are returned, they are displayed in the Streamlit UI (as tables, metrics, etc.).
- Query execution latency is measured and displayed.
- If errors occur (either in generation or execution), a friendly error is shown to the user.

### 7. (Optional) Handcrafted Query Mapping

- For specific well-known queries, a manual mapping (`query_mapping`) is checked before invoking the LLM. This is useful for optimizing common or ambiguous requests.

---

**Summary:**  
The user describes their reporting need in plain language. The app dynamically inspects the current database schema, forms a detailed prompt for the LLM, generates SQL code, adapts it for SQLite, and runs it—showing the user the results, all through an interactive web interface.

For more details, see the source code for [`mainui.py`](https://github.com/szanislaw/fcs-entreport/blob/main/mainui.py).

---
## Multi-User Support via Dynamic Schema Injection

A key feature of this engine is its **dynamic schema injection** capability, which is fundamental for multi-user scenarios:

### How Dynamic Schema Injection Enables Multi-User Support

- **Per-Session Database Context:**  
  Each time a user submits a natural language query, the application dynamically inspects the current state of the SQLite database. It retrieves the live schema (tables and columns) at that moment.

- **Prompt Customization:**  
  The system then generates a prompt for the SQLCoder LLM that includes *the exact database schema as it exists for that user/session*—not a static or hard-coded schema. This means the LLM is always given the most accurate context in which to generate SQL queries.

- **Concurrency and Isolation:**  
  Because the schema is injected into the prompt per request, different users (or sessions) can interact with different database states (e.g., if using separate databases per user or session, or if the schema is modified at runtime).  
  - If deployed in a multi-tenant setup (one database per user or organization), each user’s queries are generated and executed in the context of their own schema.
  - If the schema changes (e.g., new tables/columns added), the prompt immediately reflects those changes.

- **Example Flow:**
  1. User A and User B each access the application (potentially with different databases or after schema modifications).
  2. When User A submits a question, the application queries their database schema and injects it into the LLM prompt.
  3. When User B submits a question, the same process occurs independently for their schema.
  4. The LLM generates SQL that is always valid for the specific schema being queried.

### Why This Matters

- **No Cross-User Leakage:**  
  Queries for one user are never generated using another user’s schema, preventing confusion and potential data/security issues.
- **Extremely Flexible:**  
  The app supports rapidly evolving schemas, multi-tenant architectures, and even runtime schema modifications without any code changes or model retraining.
- **Scalable:**  
  This approach allows the engine to be used for SaaS products, internal tools, or any scenario where users may have unique or changing database structures.

### Technical Note

The dynamic schema injection is handled by the function:

```python
def get_dynamic_schema_prompt(conn: sqlite3.Connection, question: str) -> str:
    # Inspects the live database schema, formats it, and injects it into the prompt
```

This ensures every LLM prompt is tailored to the *actual* database structure at query time.

---

In summary: **Multiple users can use this engine simultaneously and safely, thanks to dynamic schema injection that personalizes every query to each user’s live database context.**

---

## File Structure

```
fcs-entreport/
│
├── mainui.py           # Main Streamlit application and backend logic
├── hotel.sql           # Database schema for demo hotel management system
├── hotel.db            # (Generated) SQLite database
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── ...                 # Additional modules/resources
```

## Setup & Usage

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- Sufficient system memory (for LLM inference)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/szanislaw/fcs-entreport.git
   cd fcs-entreport
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run mainui.py
```
- The app will open in your browser.
- On first launch, the model and tokenizer are downloaded; the demo database is initialized if needed.

## Technical Details

### Model Handling

- Uses HuggingFace's `AutoTokenizer` and `AutoModelForCausalLM` for the `defog/sqlcoder-7b-2` model.
- Loads model with float16 precision for efficiency.
- Automatically selects device (`cuda` if available, else `cpu`).

### Database Schema

- The hotel schema includes tables for regions, hotels, rooms, guests, bookings, payments, staff, services, reviews, shifts, complaints, training, rosters, and more.
- Schema is read from `hotel.sql` and loaded if tables are missing.

### SQL Conversion

- The `pg_to_sqlite` function adapts PostgreSQL SQL to SQLite, handling:
  - Schema prefix removal
  - Boolean value conversion
  - Type cast removal
  - `ILIKE` to `LOWER(col) LIKE ...`
  - Timestamp and other common differences

## Extending

- To support additional database backends, expand the SQL conversion logic or provide alternate connectors.
- For new report types or UI controls, extend the Streamlit front-end and relevant backend processing.

## License

MIT License

---

> **Note:** This README is based on analysis of [`mainui.py`](https://github.com/szanislaw/fcs-entreport/blob/bb27fa59e792ddd5c7297534785eb76010c7b16f/mainui.py). For the full, up-to-date code, see the [GitHub repository](https://github.com/szanislaw/fcs-entreport).
