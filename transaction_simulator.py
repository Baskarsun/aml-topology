"""
Transaction Simulator for AML Dashboard Demo

Continuously generates synthetic transactions and sends them to the inference API
to demonstrate real-time monitoring capabilities.
"""

import requests
import random
import datetime
import time
import threading
import argparse
from typing import Dict, List

API_URL = "http://localhost:5000/score/consolidate"

class TransactionSimulator:
    """Simulates continuous transaction flow for demo purposes."""
    
    def __init__(self, api_url: str = API_URL, rate: float = 2.0):
        """
        Initialize simulator.
        
        Args:
            api_url: URL of the inference API
            rate: Transactions per second
        """
        self.api_url = api_url
        self.rate = rate
        self.running = False
        self.stats = {
            'total_sent': 0,
            'success': 0,
            'errors': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'clean': 0
        }
        
        # Use fixed pool of accounts to enable pattern detection
        self.account_pool = [f"ACC_{i:04d}" for i in range(1001, 1051)]  # 50 accounts
        self.suspicious_accounts = [f"ACC_{i:04d}" for i in range(1051, 1061)]  # 10 suspicious
        self.high_risk_accounts = [f"ACC_{i:04d}" for i in range(1061, 1071)]  # 10 high-risk
        
        # Track transaction history for creating patterns
        self.transaction_history = []
    
    def generate_transaction(self, risk_profile: str = "normal", target_account: str = None) -> Dict:
        """
        Generate synthetic transaction with specified risk profile.
        
        Args:
            risk_profile: "normal", "suspicious", or "high_risk"
            target_account: Optional target account for transactions
        
        Returns:
            Dict with transaction features
        """
        base_transaction = {
            "amount": round(random.uniform(10, 5000), 2),
            "mcc": random.choice(["5411", "6011", "4829", "5732", "7011", "5311"]),
            "payment_type": random.choice(["card", "wire", "ach", "crypto"]),
            "device_change": False,
            "ip_risk": round(random.uniform(0, 0.3), 2),
            "count_1h": random.randint(1, 3),
            "sum_24h": round(random.uniform(100, 5000), 2),
            "uniq_payees_24h": random.randint(1, 5),
            "is_international": False,
            "country": "US",
            "avg_tx_24h": round(random.uniform(100, 2000), 2),
            "velocity_score": round(random.uniform(0, 0.4), 3)
        }
        
        # Add target if specified (for graph connectivity)
        if target_account:
            base_transaction["target_account"] = target_account
        
        # Modify based on risk profile
        if risk_profile == "suspicious":
            base_transaction.update({
                "amount": round(random.uniform(3000, 10000), 2),
                "device_change": random.choice([True, False]),
                "ip_risk": round(random.uniform(0.4, 0.7), 2),
                "count_1h": random.randint(5, 12),
                "sum_24h": round(random.uniform(8000, 25000), 2),
                "uniq_payees_24h": random.randint(8, 20),
                "velocity_score": round(random.uniform(0.5, 0.75), 3),
                "is_international": random.choice([True, False]),
                "country": random.choice(["US", "UK", "CN", "RU"])
            })
        elif risk_profile == "high_risk":
            base_transaction.update({
                "amount": round(random.uniform(7000, 25000), 2),
                "device_change": True,
                "ip_risk": round(random.uniform(0.7, 1.0), 2),
                "count_1h": random.randint(8, 20),
                "sum_24h": round(random.uniform(15000, 50000), 2),
                "uniq_payees_24h": random.randint(15, 40),
                "velocity_score": round(random.uniform(0.75, 1.0), 3),
                "is_international": True,
                "country": random.choice(["CN", "RU", "NG", "PK"]),
                "payment_type": "crypto"
            })
        
        return base_transaction
    
    def generate_events(self, risk_profile: str = "normal") -> List[str]:
        """
        Generate event sequence with specified risk profile.
        
        Args:
            risk_profile: "normal", "suspicious", or "high_risk"
        
        Returns:
            List of event strings
        """
        normal_sequences = [
            ["login_success", "view_account", "transfer", "logout"],
            ["login_success", "view_account", "logout"],
            ["login_success", "add_payee", "transfer", "logout"],
            ["login_success", "view_account", "view_account", "logout"]
        ]
        
        suspicious_sequences = [
            ["login_failed", "login_failed", "login_success", "password_change", "add_payee", "transfer", "logout"],
            ["login_success", "add_payee", "add_payee", "transfer", "max_transfer", "logout"],
            ["login_success", "password_change", "add_payee", "transfer", "transfer", "logout"]
        ]
        
        high_risk_sequences = [
            ["login_failed", "login_failed", "login_failed", "login_success", "password_change", 
             "add_payee", "add_payee", "max_transfer", "max_transfer", "logout"],
            ["login_success", "password_change", "add_payee", "add_payee", "add_payee",
             "max_transfer", "transfer", "transfer", "logout"],
            ["login_failed", "login_success", "password_change", "view_account",
             "add_payee", "max_transfer", "max_transfer", "max_transfer"]
        ]
        
        if risk_profile == "normal":
            return random.choice(normal_sequences)
        elif risk_profile == "suspicious":
            return random.choice(suspicious_sequences)
        else:
            return random.choice(high_risk_sequences)
    
    def send_transaction(self, account_id: str, risk_profile: str = "normal"):
        """Send a single transaction to the API."""
        payload = {
            "account_id": account_id,
            "transaction": self.generate_transaction(risk_profile),
            "events": self.generate_events(risk_profile)
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            self.stats['success'] += 1
            
            # Track risk levels
            risk_level = result.get('risk_level', 'CLEAN')
            if risk_level == 'HIGH':
                self.stats['high_risk'] += 1
            elif risk_level == 'MEDIUM':
                self.stats['medium_risk'] += 1
            elif risk_level == 'LOW':
                self.stats['low_risk'] += 1
            else:
                self.stats['clean'] += 1
            
            return True, result
        except Exception as e:
            self.stats['errors'] += 1
            print(f"❌ Error sending transaction: {e}")
            return False, str(e)
    
    def run(self, duration: int = None):
        """
        Run the simulator.
        
        Args:
            duration: Duration in seconds (None for infinite)
        """
        self.running = True
        start_time = time.time()
        
        print(f"🚀 Starting transaction simulator...")
        print(f"📡 API URL: {self.api_url}")
        print(f"⚡ Rate: {self.rate} transactions/second")
        print(f"⏱️  Duration: {'Infinite (Ctrl+C to stop)' if duration is None else f'{duration} seconds'}")
        print(f"👥 Account pools: {len(self.account_pool)} normal, {len(self.suspicious_accounts)} suspicious, {len(self.high_risk_accounts)} high-risk")
        print()
        
        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Determine risk profile for this transaction (60% normal, 30% suspicious, 10% high_risk)
                rand = random.random()
                if rand < 0.60:
                    risk_profile = "normal"
                    account_id = random.choice(self.account_pool)
                elif rand < 0.90:
                    risk_profile = "suspicious"
                    account_id = random.choice(self.suspicious_accounts)
                else:
                    risk_profile = "high_risk"
                    account_id = random.choice(self.high_risk_accounts)
                
                success, result = self.send_transaction(account_id, risk_profile)
                self.stats['total_sent'] += 1
                
                if success:
                    risk_score = result.get('consolidated_risk_score', 0.0)
                    risk_level = result.get('risk_level', 'CLEAN')
                    
                    # Color-coded output
                    if risk_level == 'HIGH':
                        emoji = "🔴"
                    elif risk_level == 'MEDIUM':
                        emoji = "🟡"
                    elif risk_level == 'LOW':
                        emoji = "🟢"
                    else:
                        emoji = "⚪"
                    
                    print(f"{emoji} {account_id} | Risk: {risk_score:.3f} ({risk_level}) | "
                          f"Profile: {risk_profile} | Total: {self.stats['total_sent']}")
                
                # Sleep to maintain rate
                time.sleep(1.0 / self.rate)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Simulator stopped by user")
        finally:
            self.running = False
            self.print_summary()
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("📊 SIMULATION SUMMARY")
        print("="*60)
        print(f"Total Sent:     {self.stats['total_sent']:,}")
        print(f"✅ Successful:   {self.stats['success']:,}")
        print(f"❌ Errors:       {self.stats['errors']:,}")
        print()
        print("Risk Distribution:")
        print(f"  🔴 High Risk:   {self.stats['high_risk']:,} "
              f"({100*self.stats['high_risk']/max(1,self.stats['success']):.1f}%)")
        print(f"  🟡 Medium Risk: {self.stats['medium_risk']:,} "
              f"({100*self.stats['medium_risk']/max(1,self.stats['success']):.1f}%)")
        print(f"  🟢 Low Risk:    {self.stats['low_risk']:,} "
              f"({100*self.stats['low_risk']/max(1,self.stats['success']):.1f}%)")
        print(f"  ⚪ Clean:       {self.stats['clean']:,} "
              f"({100*self.stats['clean']/max(1,self.stats['success']):.1f}%)")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AML Transaction Simulator")
    parser.add_argument('--url', type=str, default=API_URL,
                       help='API endpoint URL (default: http://localhost:5000/score/consolidate)')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='Transactions per second (default: 2.0)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in seconds (default: infinite)')
    
    args = parser.parse_args()
    
    simulator = TransactionSimulator(api_url=args.url, rate=args.rate)
    simulator.run(duration=args.duration)


if __name__ == "__main__":
    main()
