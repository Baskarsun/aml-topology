import random
import time
import string
import pandas as pd
import numpy as np

from india_aml import (
    generate_upi_transactions,
    generate_nach_data,
    generate_cin_data,
    generate_benami_transactions,
    UPIPatternDetector,
    NACHMandateAnalyzer,
    ShellCompanyCINGraph,
    BenamiIndicatorDetector,
)

class TransactionSimulator:
    def __init__(self, num_accounts=100):
        self.num_accounts = num_accounts
        self.accounts = [f"ACC_{i:04d}" for i in range(num_accounts)]
        self.transactions = []
        self.start_time = int(time.time())
        # India-specific state
        self._upi_txns: list[dict] = []
        self._nach_mandates: list[dict] = []
        self._nach_debits: list[dict] = []
        self._cin_companies: list[dict] = []
        self._benami_txns: list[dict] = []

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

    # ------------------------------------------------------------------
    # India-specific injection methods
    # ------------------------------------------------------------------

    def inject_upi_patterns(
        self,
        num_organic: int = 200,
        num_structuring_actors: int = 3,
        num_roundtrip_pairs: int = 2,
        num_mule_accounts: int = 5,
    ):
        """
        Generate synthetic UPI transactions covering structuring,
        rapid round-trips, and mule-VPA patterns.
        Stores results internally; analyse with run_india_analysis().
        """
        print(f"Injecting UPI patterns: {num_organic} organic, "
              f"{num_structuring_actors} structuring actors, "
              f"{num_roundtrip_pairs} round-trip pairs, "
              f"{num_mule_accounts} mule accounts")
        self._upi_txns = generate_upi_transactions(
            num_organic=num_organic,
            num_structuring_actors=num_structuring_actors,
            num_roundtrip_pairs=num_roundtrip_pairs,
            num_mule_accounts=num_mule_accounts,
            base_time=self.start_time - 7 * 86400,
        )

    def inject_nach_patterns(
        self,
        num_mandates: int = 30,
        num_debits_per_mandate: int = 12,
    ):
        """
        Generate synthetic NACH mandate and debit records covering
        amount escalation, mandate churn, and ghost-mandate patterns.
        """
        print(f"Injecting NACH patterns: {num_mandates} mandates, "
              f"~{num_debits_per_mandate} debits each")
        self._nach_mandates, self._nach_debits = generate_nach_data(
            num_mandates=num_mandates,
            num_debits_per_mandate=num_debits_per_mandate,
        )

    def inject_shell_company_cin(
        self,
        num_companies: int = 20,
        num_shell_clusters: int = 2,
        cluster_size: int = 4,
    ):
        """
        Generate a synthetic CIN-director dataset with embedded shell
        company clusters sharing directors and low paid-up capital.
        """
        print(f"Injecting shell CIN graph: {num_companies} organic + "
              f"{num_shell_clusters} clusters x {cluster_size} companies")
        self._cin_companies = generate_cin_data(
            num_companies=num_companies,
            num_shell_clusters=num_shell_clusters,
            cluster_size=cluster_size,
        )

    def inject_benami_patterns(
        self,
        num_organic: int = 100,
        num_benami_actors: int = 5,
    ):
        """
        Generate synthetic benami transaction records covering income
        mismatch, third-party property, round-trips, and PAN fragmentation.
        """
        print(f"Injecting benami patterns: {num_organic} organic, "
              f"{num_benami_actors} benami actors")
        self._benami_txns = generate_benami_transactions(
            num_organic=num_organic,
            num_benami_actors=num_benami_actors,
            base_time=self.start_time - 30 * 86400,
        )

    def run_india_analysis(self) -> dict:
        """
        Run all India-specific AML detectors over the injected data and
        return a structured dict of alerts keyed by signal category.
        """
        results: dict[str, list] = {}

        # UPI
        if self._upi_txns:
            det = UPIPatternDetector(self._upi_txns)
            results["upi_structuring"] = det.detect_structuring()
            results["upi_rapid_roundtrip"] = det.detect_rapid_roundtrip()
            results["upi_mule_vpa"] = det.detect_mule_vpa()

        # NACH
        if self._nach_mandates:
            ana = NACHMandateAnalyzer(self._nach_mandates, self._nach_debits)
            results["nach_amount_escalation"] = ana.detect_amount_escalation()
            results["nach_mandate_churn"] = ana.detect_mandate_churn()
            results["nach_high_failure_rate"] = ana.detect_high_failure_rate()
            results["nach_ghost_mandate"] = ana.detect_ghost_mandate()

        # CIN shell companies
        if self._cin_companies:
            cg = ShellCompanyCINGraph(self._cin_companies)
            results["cin_circular_directorship"] = cg.detect_circular_directorships()
            results["cin_director_hub"] = cg.detect_director_hubs()
            results["cin_shell_candidates"] = cg.detect_fresh_shell_candidates()
            results["cin_connected_clusters"] = cg.get_connected_components()

        # Benami
        if self._benami_txns:
            bd = BenamiIndicatorDetector(self._benami_txns)
            results["benami_income_mismatch"] = bd.detect_income_mismatch()
            results["benami_third_party_property"] = bd.detect_third_party_property()
            results["benami_round_trip"] = bd.detect_round_trip_cash()
            results["benami_pan_fragmentation"] = bd.detect_pan_mismatch_cluster()

        return results
