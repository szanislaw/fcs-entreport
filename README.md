# FCS Entreport

Enterprise Reporting using Streamlit

## Overview

FCS Entreport is an enterprise-level reporting application that leverages [Streamlit](https://streamlit.io/) for interactive dashboards and integrates AI-powered SQL generation via large language models. The application is designed to provide flexible, natural language-driven business intelligence and reporting, primarily for hotel management scenarios.

The main entry point for the application is [`mainui.py`](mainui.py).

## Key Features

- **AI-Powered SQL Generation:** Utilizes the `defog/sqlcoder-7b-2` language model (via HuggingFace Transformers) to generate SQL queries from natural language.
- **Database Agnostic:** Includes utilities to convert PostgreSQL-style SQL to SQLite-compatible queries.
- **One-Click Demo Database:** Automatically initializes a rich SQLite database (`hotel.db`) from a provided schema (`hotel.sql`) if not already present.
- **Streamlit UI:** Provides a modern, browser-based interface for interacting with data, generating reports, and visualizing results.
- **Enterprise Hotel Management Data Model:** Supports a comprehensive schema including guests, bookings, payments, staff, services, reviews, and more.

## How It Works

`mainui.py` orchestrates the following:

1. **Model Initialization**
   - Loads the `defog/sqlcoder-7b-2` LLM and tokenizer using HuggingFace Transformers.
   - Model is loaded onto GPU (CUDA) if available, or CPU otherwise.
   - Caching is used to avoid repeated model load times.

2. **Database Bootstrapping**
   - Connects to `hotel.db` (SQLite).
   - Checks for the presence of key tables (e.g., `guests`, `bookings`, `payments`).
   - If missing, executes the SQL schema from `hotel.sql` to initialize the database.

3. **PostgreSQL to SQLite SQL Conversion**
   - Contains a utility (`pg_to_sqlite`) to adapt SQL queries written in PostgreSQL dialect (e.g., with `ILIKE`, `TRUE`/`FALSE`, schema prefixes, typecasts) to SQLite syntax.

4. **Streamlit Frontend**
   - Uses Streamlit widgets for user interaction (not fully shown in the snippet, but expected for report parameters, SQL input, model outputs, etc.).
   - Enables users to input natural language questions, generates SQL with the LLM, executes them against the SQLite database, and displays results.

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
