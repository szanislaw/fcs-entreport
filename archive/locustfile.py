from locust import HttpUser, task, between
import random

SAMPLE_QUERIES = [
    "How many rooms were cleaned yesterday?",
    "Show me top 3 staff by rooms cleaned.",
    "How many guest complaints this month?",
    "What is the inspection pass rate?",
    "List bookings in July."
]

class StreamlitUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask_question(self):
        question = random.choice(SAMPLE_QUERIES)
        self.client.get("/", name="Home Page")
        self.client.post("/", data={"user_query": question}, name="Ask Question")
