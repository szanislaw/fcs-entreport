import sqlite3
import re
from datetime import datetime

input_file = 'job_detail_listing.sql'
output_db = 'job_detail_listing.db'

conn = sqlite3.connect(output_db)
cur = conn.cursor()

# Drop and recreate the table
cur.execute('DROP TABLE IF EXISTS job_detail_listing')
cur.execute("""
CREATE TABLE job_detail_listing (
    department_name TEXT,
    date_time_created_ DATETIME,
    job_status TEXT,
    job_order TEXT,
    guest_name TEXT,
    location TEXT,
    service_item_category TEXT,
    service_item TEXT,
    quantity TEXT,
    remarks TEXT,
    date_time_deadline_ DATETIME,
    date_time_completed_ DATETIME,
    escalation_level TEXT,
    created_by_department_ TEXT,
    created_by_user_ TEXT,
    assigned_to_department_ TEXT,
    assigned_to_user_ TEXT,
    acknowledged_by_department_ TEXT,
    acknowledged_by_user_ TEXT,
    completed_by_department_ TEXT,
    completed_by_user_ TEXT,
    escalated_to TEXT,
    comments TEXT
)
""")

def convert_date(s):
    if not s:
        return None
    s = s.strip().strip("'")
    if s.lower() in ['none', '', 'null']:
        return None
    try:
        return datetime.strptime(s, '%d %b %Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
    except:
        try:
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        except:
            return s

def parse_sql_values(raw):
    """
    Parses a SQL VALUES(...) string into a Python list.
    Handles single quotes, escaped quotes (''), and multi-line fields.
    """
    raw = raw.strip()
    if raw.endswith(';'):
        raw = raw[:-1]
    placeholder = '␟'  # Temporary replacement for SQL-escaped single quote
    raw = raw.replace("''", placeholder)

    result = []
    current = ''
    in_quote = False
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == "'" and not in_quote:
            in_quote = True
            i += 1
            continue
        elif char == "'" and in_quote:
            in_quote = False
            result.append(current.replace(placeholder, "'").strip())
            current = ''
            # Skip comma if exists
            i += 1
            while i < len(raw) and raw[i] in ' ,\n':
                i += 1
            continue
        elif in_quote:
            current += char
        else:
            # For unquoted NULLs
            if raw[i:i+4].lower() == 'null':
                result.append(None)
                i += 4
                while i < len(raw) and raw[i] in ' ,\n':
                    i += 1
                continue
        i += 1
    return result

# Read full .sql content
with open(input_file, 'r', encoding='utf-8') as f:
    full_sql = f.read()

# Match INSERT statements
pattern = re.compile(r'INSERT INTO job_detail_listing VALUES\s*\((.*?)\);', re.DOTALL)
matches = pattern.findall(full_sql)

inserted, skipped = 0, 0

for raw in matches:
    try:
        values = parse_sql_values(raw)

        if len(values) != 23:
            print(f"⚠️ Skipped row with {len(values)} values: {values[:3]}...")
            skipped += 1
            continue

        for i in [1, 10, 11]:
            values[i] = convert_date(values[i])

        placeholders = ','.join(['?'] * 23)
        cur.execute(f"INSERT INTO job_detail_listing VALUES ({placeholders})", values)
        inserted += 1

    except Exception as e:
        print(f"❌ Error parsing row: {e}")
        skipped += 1

conn.commit()
conn.close()

print(f"\n✅ Import complete: {inserted} inserted | {skipped} skipped.")
