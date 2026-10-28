import json
import random
import time
from datetime import datetime, timezone
from faker import Faker
from textblob import TextBlob

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

def stream_data(filename):
    try:
        with open(filename, 'a') as f:
            while True:
                post = generate_post()
                f.write(json.dumps(post) + '\n')
                f.flush()
                time.sleep(random.uniform(0.3, 1.5))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    open("social_stream.jsonl", "w").close()
    stream_data("social_stream.jsonl")