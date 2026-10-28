# data_generator.py
import json
import random
import time
from datetime import datetime, timezone
from faker import Faker
from textblob import TextBlob
import threading
import os

fake = Faker()
platforms = ["X", "Reddit", "LinkedIn", "Instagram"]
topics = ["new product", "customer service", "update", "pricing", "downtime", "UI"]
adjectives = ["terrible", "amazing", "worst", "fantastic", "mediocre", "awful", "brilliant", "okay", "unusable", "perfect"]

def generate_post():
    topic = random.choice(topics)
    adj = random.choice(adjectives)
    base_text = fake.sentence(nb_words=8)
    text = f"The {topic} is {adj}. {base_text}"
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": fake.user_name(),
        "platform": random.choice(platforms),
        "text": text,
        "polarity": polarity,
        "sentiment": label
    }

def _stream_loop(filename):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            pass
            
    while True:
        post = generate_post()
        with open(filename, 'a') as f:
            f.write(json.dumps(post) + '\n')
            f.flush()
        time.sleep(random.uniform(0.3, 1.5))

def start_background_stream(filename="social_stream.jsonl"):
    # Runs the generator in a background daemon thread
    thread = threading.Thread(target=_stream_loop, args=(filename,), daemon=True)
    thread.start()
