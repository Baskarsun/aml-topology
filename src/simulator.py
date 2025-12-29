import random
import time
import pandas as pd
import numpy as np

class TransactionSimulator:
    def __init__(self, num_accounts=100):
        self.num_accounts = num_accounts
        self.accounts = [f"ACC_{i:04d}" for i in range(num_accounts)]
        self.transactions = []
        self.start_time = int(time.time())

    def _get_random_timestamp(self, time_window=86400):
        """Returns a timestamp within the given window from start_time."""
        offset = random.randint(0, time_window)
        return self.start_time + offset

    def generate_organic_traffic(self, num_transactions=500):
        """Generates random background noise transactions."""
        print(f"Generating {num_transactions} organic transactions...")
        for _ in range(num_transactions):
            src = random.choice(self.accounts)
            dst = random.choice(self.accounts)
            while src == dst:
                dst = random.choice(self.accounts)
            
            amount = round(random.uniform(10.0, 500.0), 2)
            timestamp = self._get_random_timestamp()
            
            self.transactions.append({
                "source": src,
                "target": dst,
                "amount": amount,
                "timestamp": timestamp,
                "type": "organic"
            })

    def inject_fan_in(self, hub_account, num_spokes=10, avg_amount=9000, variance=500):
        """Simulates a Fan-In (Structuring/Placement) pattern."""
        print(f"Injecting Fan-In pattern: {num_spokes} spokes -> {hub_account}")
        spokes = random.sample([a for a in self.accounts if a != hub_account], num_spokes)
        
        # Tight timeframe for coordinated activity
        base_time = self._get_random_timestamp(time_window=3600) 

        for spoke in spokes:
            amount = round(random.gauss(avg_amount, variance), 2)
            self.transactions.append({
                "source": spoke,
                "target": hub_account,
                "amount": amount,
                "timestamp": base_time + random.randint(0, 600), # Within 10 mins
                "type": "launder_fan_in"
            })

    def inject_fan_out(self, hub_account, num_beneficiaries=10, total_amount=100000):
        """Simulates a Fan-Out (Integration/Payroll) pattern."""
        print(f"Injecting Fan-Out pattern: {hub_account} -> {num_beneficiaries} beneficiaries")
        beneficiaries = random.sample([a for a in self.accounts if a != hub_account], num_beneficiaries)
        
        amount_per_ben = total_amount / num_beneficiaries
        base_time = self._get_random_timestamp(time_window=3600)

        for ben in beneficiaries:
            # Slight variance to look less mechanical, but still highly suspicious
            amount = round(amount_per_ben * random.uniform(0.95, 1.05), 2)
            self.transactions.append({
                "source": hub_account,
                "target": ben,
                "amount": amount,
                "timestamp": base_time + random.randint(0, 600),
                "type": "launder_fan_out"
            })

    def inject_cycle(self, length=4, amount=50000):
        """Simulates a Layering Cycle: A -> B -> C -> D -> A."""
        cycle_nodes = random.sample(self.accounts, length)
        print(f"Injecting Cycle pattern: {' -> '.join(cycle_nodes)} -> {cycle_nodes[0]}")
        
        base_time = self._get_random_timestamp(time_window=10000)
        
        for i in range(length):
            src = cycle_nodes[i]
            dst = cycle_nodes[(i + 1) % length] # Wrap around to start
            
            # Fee shrinkage (money mules take a cut)
            current_amount = amount * (0.98 ** i) 
            
            self.transactions.append({
                "source": src,
                "target": dst,
                "amount": round(current_amount, 2),
                "timestamp": base_time + (i * 1800), # 30 mins delay between hops
                "type": "launder_cycle"
            })

    def get_dataframe(self):
        return pd.DataFrame(self.transactions)
