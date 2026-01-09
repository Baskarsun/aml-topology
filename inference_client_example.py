import requests
import random
import datetime

API_URL = "http://localhost:5000/score/consolidate"

# Generate synthetic transaction features
def generate_synthetic_transaction():
    return {
        "account_id": f"ACC_{random.randint(1000, 9999)}",
        "amount": round(random.uniform(10, 5000), 2),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "counterparty_id": f"ACC_{random.randint(1000, 9999)}",
        "location": random.choice(["London", "New York", "Singapore", "Berlin"]),
        "transaction_type": random.choice(["wire", "card", "cash", "crypto"]),
        "mcc": random.choice([5411, 6011, 4829, 5732]),
        "device_id": f"DEV_{random.randint(100, 999)}",
        "ip_address": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
        "is_international": random.choice([True, False]),
        "velocity_score": round(random.uniform(0, 1), 3),
        "avg_tx_24h": round(random.uniform(10, 2000), 2),
        "uniq_payees_24h": random.randint(1, 10),
        "sum_24h": round(random.uniform(10, 10000), 2),
        "count_1h": random.randint(1, 5),
        "device_change": random.choice([True, False]),
        "ip_risk": round(random.uniform(0, 1), 2)
    }

# Generate synthetic event sequence
def generate_synthetic_events():
    event_types = [
        "login_success", "login_failed", "password_change", "add_payee",
        "view_account", "transfer", "max_transfer", "logout"
    ]
    return random.choices(event_types, k=random.randint(5, 15))

def main():
    payload = {
        "account_id": f"ACC_{random.randint(1000, 9999)}",
        "transaction": generate_synthetic_transaction(),
        "events": generate_synthetic_events()
    }
    print("Sending payload:")
    print(payload)
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print("\nJSON response from server:")
        print(response.json())
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
