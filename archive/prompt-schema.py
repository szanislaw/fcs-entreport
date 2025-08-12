from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load SQLCoder-7B-2
tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", device_map="auto", torch_dtype=torch.float16)

# Define prompt schema
schema_prompt = """
### Task
Generate a SQL query to answer the following question:
{question}

### Database Schema
CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name TEXT);
CREATE TABLE hotels (hotel_id INTEGER PRIMARY KEY, region_id INTEGER, hotel_name TEXT, city TEXT);
CREATE TABLE rooms (room_id INTEGER PRIMARY KEY, hotel_id INTEGER, room_type TEXT, base_price REAL);
CREATE TABLE guests (guest_id INTEGER PRIMARY KEY, guest_name TEXT, nationality TEXT);
CREATE TABLE bookings (booking_id INTEGER PRIMARY KEY, room_id INTEGER, check_in TEXT, check_out TEXT, guest_id INTEGER, total_price REAL);

### SQL
Given the above schema, here is the SQL query:
"""
