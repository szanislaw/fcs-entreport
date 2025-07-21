
import sqlite3

# === Load SQL from file ===
with open("hotel_operations.sql", "r") as file:
    sql_script = file.read()

# === Create in-memory SQLite database ===
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# === Execute the SQL script ===
cursor.executescript(sql_script)

# === Sample Analyses ===

# 1. Total revenue per region on 2025-07-20
print("🔹 Revenue by Region on 2025-07-20")
query1 = """
SELECT 
    r.region_name,
    SUM(b.total_price) AS total_revenue
FROM bookings b
JOIN rooms rm ON b.room_id = rm.room_id
JOIN hotels h ON rm.hotel_id = h.hotel_id
JOIN regions r ON h.region_id = r.region_id
WHERE date('2025-07-20') BETWEEN b.check_in AND DATE(b.check_out, '-1 day')
GROUP BY r.region_name;
"""
for row in cursor.execute(query1):
    print(row)

# 2. Number of bookings per hotel
print("\n🔹 Bookings per Hotel")
query2 = """
SELECT h.hotel_name, COUNT(*) AS num_bookings
FROM bookings b
JOIN rooms r ON b.room_id = r.room_id
JOIN hotels h ON r.hotel_id = h.hotel_id
GROUP BY h.hotel_name;
"""
for row in cursor.execute(query2):
    print(row)

# 3. Top guests by total spend
print("\n🔹 Top Guests by Spend")
query3 = """
SELECT g.guest_name, SUM(b.total_price) AS total_spend
FROM bookings b
JOIN guests g ON b.guest_id = g.guest_id
GROUP BY g.guest_name
ORDER BY total_spend DESC;
"""
for row in cursor.execute(query3):
    print(row)

# Close the connection
conn.close()
