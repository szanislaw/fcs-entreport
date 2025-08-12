import sqlite3, os, shutil

SRC = "schemas/cleaning.db"
DST = "schemas/cleaning_simplified.db"

# Safety: start fresh
if os.path.exists(DST):
    os.remove(DST)

# Connect
src = sqlite3.connect(SRC)
dst = sqlite3.connect(DST)

# Pragmas
for c in (src, dst):
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA foreign_keys=ON;")

cur_s = src.cursor()
cur_d = dst.cursor()

# --- Create simplified schema ---
cur_d.executescript("""
BEGIN;

-- Dimensions
CREATE TABLE property (
  property_uuid TEXT PRIMARY KEY
);

CREATE TABLE "user" (
  user_uuid TEXT PRIMARY KEY,
  display_name TEXT
);

CREATE TABLE location (
  location_uuid TEXT PRIMARY KEY
);

-- Service types (from hskp_service_type)
CREATE TABLE service_type (
  type_uuid TEXT PRIMARY KEY,
  property_uuid TEXT REFERENCES property(property_uuid),
  type_code TEXT UNIQUE,
  type_name TEXT,
  duration INTEGER,
  credit INTEGER,
  inspection INTEGER,             -- 0/1
  inspection_duration INTEGER,
  inspection_credit INTEGER,
  priority INTEGER,
  active INTEGER,                 -- 0/1
  created_date TEXT,
  modified_date TEXT
);

-- Core cleaning order (from hskp_cleaning_order)
CREATE TABLE cleaning_order (
  cleaning_uuid TEXT PRIMARY KEY,
  property_uuid TEXT REFERENCES property(property_uuid),
  cleaning_no TEXT,
  location_uuid TEXT REFERENCES location(location_uuid),
  service_type_code TEXT REFERENCES service_type(type_code),
  credit REAL,
  duration INTEGER,
  status INTEGER,
  priority INTEGER,
  inspection INTEGER,             -- 0/1
  start_time TEXT,
  created_date TEXT,
  created_by TEXT,
  modified_date TEXT,
  modified_by TEXT,
  assigned_uuid TEXT,
  assigned_name TEXT,
  acknowledged_uuid TEXT,
  acknowledged_name TEXT,
  completed_uuid TEXT,
  completed_name TEXT,
  completed_date TEXT
);

-- Per-user actions for a cleaning (from hskp_cleaning_order_detail)
CREATE TABLE cleaning_order_detail (
  detail_uuid TEXT PRIMARY KEY,
  cleaning_uuid TEXT REFERENCES cleaning_order(cleaning_uuid) ON DELETE CASCADE,
  status INTEGER,
  user_uuid TEXT REFERENCES "user"(user_uuid),
  start_time TEXT,
  end_time TEXT,
  credit REAL
);

-- Helpful indexes
CREATE INDEX idx_service_type_code ON service_type(type_code);
CREATE INDEX idx_cleaning_order_service_type_code ON cleaning_order(service_type_code);
CREATE INDEX idx_cleaning_order_location ON cleaning_order(location_uuid);
CREATE INDEX idx_cleaning_order_created ON cleaning_order(created_date);
CREATE INDEX idx_detail_cleaning ON cleaning_order_detail(cleaning_uuid);
CREATE INDEX idx_detail_user ON cleaning_order_detail(user_uuid);

COMMIT;
""")

# --- Helpers ---
def to_int_bool(val):
    if val is None: return None
    if isinstance(val, (int, float)): return 1 if val != 0 else 0
    s = str(val).strip().lower()
    if s in ("1","true","t","yes","y"): return 1
    if s in ("0","false","f","no","n",""): return 0
    # fallback: treat anything else as 1
    return 1

# --- Populate dimensions ---

# property_uuid from hskp_service_type and hskp_cleaning_order
for (tbl, col) in [
    ("hskp_service_type", "property_uuid"),
    ("hskp_cleaning_order", "property_uuid"),
]:
    try:
        for (pid,) in cur_s.execute(f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL"):
            cur_d.execute("INSERT OR IGNORE INTO property(property_uuid) VALUES (?)", (pid,))
    except sqlite3.Error:
        pass

# location
try:
    for (lid,) in cur_s.execute("SELECT DISTINCT location_uuid FROM hskp_cleaning_order WHERE location_uuid IS NOT NULL"):
        cur_d.execute("INSERT OR IGNORE INTO location(location_uuid) VALUES (?)", (lid,))
except sqlite3.Error:
    pass

# users: collect from various *_uuid/name columns we can see
user_sources = [
    ("hskp_cleaning_order_detail","user_uuid", None),
    ("hskp_cleaning_order","assigned_uuid","assigned_name"),
    ("hskp_cleaning_order","acknowledged_uuid","acknowledged_name"),
    ("hskp_cleaning_order","completed_uuid","completed_name"),
]
for tbl, uuid_col, name_col in user_sources:
    try:
        if name_col:
            rows = cur_s.execute(f"""
                SELECT DISTINCT {uuid_col}, {name_col}
                FROM {tbl}
                WHERE {uuid_col} IS NOT NULL
            """).fetchall()
            for uid, name in rows:
                cur_d.execute('INSERT OR IGNORE INTO "user"(user_uuid, display_name) VALUES (?,?)',
                              (uid, name))
        else:
            for (uid,) in cur_s.execute(f"SELECT DISTINCT {uuid_col} FROM {tbl} WHERE {uuid_col} IS NOT NULL"):
                cur_d.execute('INSERT OR IGNORE INTO "user"(user_uuid) VALUES (?)', (uid,))
    except sqlite3.Error:
        pass

dst.commit()

# --- Migrate service_type ---
try:
    for row in cur_s.execute("""
        SELECT
          type_uuid, property_uuid, type_code, type_name, duration, credit,
          inspection, inspection_duration, inspection_credit, priority, active,
          created_date, modified_date
        FROM hskp_service_type
    """):
        (type_uuid, property_uuid, type_code, type_name, duration, credit,
         inspection, inspection_duration, inspection_credit, priority, active,
         created_date, modified_date) = row

        cur_d.execute("""
            INSERT OR REPLACE INTO service_type(
              type_uuid, property_uuid, type_code, type_name, duration, credit,
              inspection, inspection_duration, inspection_credit, priority, active,
              created_date, modified_date
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            type_uuid, property_uuid, type_code, type_name,
            int(duration) if duration is not None else None,
            int(credit) if credit is not None else None,
            to_int_bool(inspection),
            int(inspection_duration) if inspection_duration is not None else None,
            int(inspection_credit) if inspection_credit is not None else None,
            int(priority) if priority is not None else None,
            to_int_bool(active),
            created_date, modified_date
        ))
except sqlite3.Error:
    pass

dst.commit()

# --- Migrate cleaning_order ---
try:
    for row in cur_s.execute("""
        SELECT
          cleaning_uuid, property_uuid, cleaning_no, location_uuid, service_type,
          credit, duration, status, priority, inspection, start_time,
          created_date, created_by, modified_date, modified_by,
          assigned_uuid, assigned_name,
          acknowledged_uuid, acknowledged_name,
          completed_uuid, completed_name, completed_date
        FROM hskp_cleaning_order
    """):
        cur_d.execute("""
            INSERT OR REPLACE INTO cleaning_order(
              cleaning_uuid, property_uuid, cleaning_no, location_uuid,
              service_type_code, credit, duration, status, priority, inspection,
              start_time, created_date, created_by, modified_date, modified_by,
              assigned_uuid, assigned_name, acknowledged_uuid, acknowledged_name,
              completed_uuid, completed_name, completed_date
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row[0], row[1], row[2], row[3],
            row[4],           # service_type_code (originally text code)
            row[5], int(row[6]) if row[6] is not None else None,
            int(row[7]) if row[7] is not None else None,
            int(row[8]) if row[8] is not None else None,
            to_int_bool(row[9]),
            row[10], row[11], row[12], row[13], row[14],
            row[15], row[16],
            row[17], row[18],
            row[19], row[20], row[21]
        ))
except sqlite3.Error:
    pass

dst.commit()

# --- Migrate cleaning_order_detail ---
try:
    for row in cur_s.execute("""
        SELECT detail_uuid, cleaning_uuid, status, user_uuid, start_time, end_time, credit
        FROM hskp_cleaning_order_detail
    """):
        (detail_uuid, cleaning_uuid, status, user_uuid, start_time, end_time, credit) = row
        cur_d.execute("""
            INSERT OR REPLACE INTO cleaning_order_detail(
              detail_uuid, cleaning_uuid, status, user_uuid, start_time, end_time, credit
            ) VALUES (?,?,?,?,?,?,?)
        """, (
            detail_uuid, cleaning_uuid,
            int(status) if status is not None else None,
            user_uuid, start_time, end_time, credit
        ))
except sqlite3.Error:
    pass

dst.commit()

# --- Convenience analytics views ---
cur_d.executescript("""
CREATE VIEW v_cleaning_order_enriched AS
SELECT
  co.cleaning_uuid,
  co.cleaning_no,
  co.created_date,
  co.completed_date,
  co.status,
  co.priority,
  co.credit,
  co.duration,
  st.type_code,
  st.type_name,
  st.inspection AS service_requires_inspection,
  co.location_uuid,
  co.assigned_uuid,
  co.assigned_name
FROM cleaning_order co
LEFT JOIN service_type st
  ON co.service_type_code = st.type_code;

-- Attendant throughput by day
CREATE VIEW v_attendant_daily_jobs AS
SELECT
  date(co.created_date) AS job_date,
  co.assigned_uuid,
  co.assigned_name,
  COUNT(*) AS jobs_assigned,
  SUM(COALESCE(co.credit,0)) AS total_credit
FROM cleaning_order co
GROUP BY 1,2,3;

-- Who worked on what (from detail table)
CREATE VIEW v_detail_work AS
SELECT
  d.user_uuid,
  u.display_name,
  d.cleaning_uuid,
  d.status,
  d.start_time,
  d.end_time,
  d.credit
FROM cleaning_order_detail d
LEFT JOIN "user" u ON u.user_uuid = d.user_uuid;
""")

dst.commit()

# Optimize
cur_d.execute("PRAGMA optimize;")
dst.commit()

src.close()
dst.close()

print(f"Done. Wrote: {DST}")
