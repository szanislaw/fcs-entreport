# sql_fixes.py

import re

def apply_sql_fixes(sql: str) -> str:
    # Fix ILIKE → LOWER(...) LIKE
    sql = re.sub(
        r"(\w+(?:\.\w+)?)\s+ILIKE\s+'(.*?)'",
        lambda m: f"LOWER({m.group(1)}) LIKE '%{m.group(2).lower()}%'",
        sql,
        flags=re.IGNORECASE
    )

    # Fix EXTRACT(YEAR FROM col)
    sql = re.sub(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%Y', \1) AS INTEGER)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"EXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%m', \1) AS INTEGER)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"EXTRACT\s*\(\s*DAY\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%d', \1) AS INTEGER)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"EXTRACT\s*\(\s*DOW\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%w', \1) AS INTEGER)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"EXTRACT\s*\(\s*HOUR\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%H', \1) AS INTEGER)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"EXTRACT\s*\(\s*MINUTE\s+FROM\s+([^)]+?)\s*\)", r"CAST(strftime('%M', \1) AS INTEGER)", sql, flags=re.IGNORECASE)

    # Fix PostgreSQL-style casting
    sql = re.sub(r"(\w+(?:\.\w+)?)::(\w+)", r"CAST(\1 AS \2)", sql)

    # Fix date_trunc and to_timestamp combinations
    sql = re.sub(r"date_trunc\s*\(\s*'day'\s*,\s*to_timestamp\s*\(([^)]+)\)\s*\)", r"date(substr(\1, 1, 10))", sql, flags=re.IGNORECASE)
    sql = re.sub(r"strftime\(\s*'(%[YmdwHMS])'\s*,\s*to_timestamp\(([^)]+)\)\s*\)", r"strftime('\1', \2)", sql, flags=re.IGNORECASE)
    sql = re.sub(r"to_timestamp\s*\(([^)]+)\)", r"\1", sql, flags=re.IGNORECASE)
    sql = re.sub(r"date_trunc\s*\(\s*'day'\s*,\s*([^)]+?)\s*\)", r"date(substr(\1, 1, 10))", sql, flags=re.IGNORECASE)
    sql = re.sub(r"date_trunc\s*\(\s*'month'\s*,\s*([^)]+?)\s*\)", r"date(substr(\1, 1, 7) || '-01')", sql, flags=re.IGNORECASE)

    # Fix date_part
    sql = re.sub(r"date_part\s*\(\s*'year'\s*,\s*([^)]+?)\s*\)", r"CAST(strftime('%Y', \1) AS INTEGER)", sql, flags=re.IGNORECASE)

    # Fix INTERVAL expressions
    sql = re.sub(
        r"(CURRENT_DATE|CURRENT_TIMESTAMP)\s*-\s*INTERVAL\s*'(\d+)\s+(day|days|week|weeks|month|months|year|years)'",
        lambda m: f"{'date' if m.group(1).upper() == 'CURRENT_DATE' else 'datetime'}('now', '-{m.group(2)} {m.group(3)}')",
        sql,
        flags=re.IGNORECASE
    )

    # Fix datetime subtraction
    sql = re.sub(r"\b([a-zA-Z_][\w\.]*)\s*-\s*([a-zA-Z_][\w\.]*)\b", r"julianday(\1) - julianday(\2)", sql)

    return sql
