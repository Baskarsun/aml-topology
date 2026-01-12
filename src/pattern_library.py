"""
AML Pattern Library
====================
Comprehensive library of money laundering patterns for synthetic training data generation.
Implements patterns from "The 2025 Horizon" report including:
- Peel Chains with Wash Trading camouflage
- Nested Cycles for tracer confusion
- Cuckoo Smurfing exits
- And more adversarial patterns

Reference: Section 6.3 "The Algorithmic Peel-Wash"
"""

import random
import time
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional


class PatternLibrary:
    """
    Library of AML patterns for generating synthetic training data.
    Each pattern returns the list of nodes created.
    """
    
    def __init__(self, base_time: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize the pattern library.
        
        Args:
            base_time: Base timestamp for temporal patterns (default: now)
            seed: Random seed for reproducibility
        """
        self.base_time = base_time or int(time.time())
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # ========================================================================
    # PEEL CHAIN PATTERNS (Section 6.3)
    # ========================================================================
    
    def inject_sophisticated_peel_chain(
        self, 
        G: nx.DiGraph, 
        existing_nodes: List[str], 
        length: int = 15, 
        initial_amount: float = 100000,
        start_time: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Generates an 'Algorithmic Peel-Wash' chain with temporal realism, 
        nested cycles, and cuckoo exits.
        
        Reference: Section 6.3 'The Algorithmic Peel-Wash'
        
        Args:
            G: NetworkX graph to add nodes/edges to
            existing_nodes: List of existing account node IDs
            length: Number of hops in the peel chain
            initial_amount: Starting amount to launder
            start_time: Base timestamp for the chain
            
        Returns:
            Dict with keys: 'chain_nodes', 'wash_nodes', 'cycle_nodes', 'exit_nodes'
        """
        if start_time is None:
            start_time = self.base_time
        
        current_node = random.choice(existing_nodes)
        current_amt = initial_amount
        t = start_time
        
        chain_nodes = []
        wash_nodes = []
        cycle_nodes = []
        exit_nodes = []
        
        for i in range(length):
            if current_amt <= 100:
                break
            
            # --- 1. Temporal Logic: Rapid Fire vs Evasion ---
            # 80% chance of 'Rapid Fire' (minutes), 20% chance of 'Pause' (days)
            if random.random() < 0.8:
                t += random.randint(60, 600)  # 1-10 mins
            else:
                t += random.randint(86400, 172800)  # 1-2 days
                
            # --- 2. Create Next Hop (The Mule) ---
            next_node = f"Chain_{start_time}_{i}"
            G.add_node(next_node, ntype='account', label=1, pattern='peel_chain')
            chain_nodes.append(next_node)
            
            # --- 3. The Peel (Variable Decay) ---
            # Variable peel amount (3-8%) to avoid static rules
            peel_amt = current_amt * random.uniform(0.03, 0.08)
            current_amt -= peel_amt
            
            # Main chain transaction
            G.add_edge(current_node, next_node, 
                       amount=round(current_amt, 2), 
                       timestamp=t, 
                       type='peel_chain')
            
            # --- 4. Wash Trading (The Camouflage) ---
            # "Algorithmic Peel-Wash": Simultaneous high-freq noise
            num_wash = random.randint(2, 5)
            for j in range(num_wash):
                w_node = f"Wash_{start_time}_{i}_{random.randint(100, 999)}"
                G.add_node(w_node, ntype='account', label=1, pattern='wash_bot')
                wash_nodes.append(w_node)
                
                # Wash pattern: A -> W -> A (within seconds of main tx)
                wash_amt = peel_amt * random.uniform(0.1, 0.4)
                G.add_edge(current_node, w_node, 
                           amount=round(wash_amt, 2), 
                           timestamp=t + random.randint(1, 5), 
                           type='wash_trade')
                G.add_edge(w_node, current_node, 
                           amount=round(wash_amt * 0.99, 2), 
                           timestamp=t + random.randint(6, 10), 
                           type='wash_return')
                           
            # --- 5. Nested Cycle Injection (The Trap) ---
            # "Nested Cycles": Create a loop to confuse tracers (30% chance after hop 2)
            if i > 2 and random.random() < 0.3:
                cycle_node = f"Cycle_{start_time}_{i}"
                G.add_node(cycle_node, ntype='account', label=1, pattern='layering_cycle')
                cycle_nodes.append(cycle_node)
                
                # A -> Cycle -> A
                G.add_edge(current_node, cycle_node, 
                           amount=round(current_amt * 0.1, 2), 
                           timestamp=t + 1, 
                           type='layering_cycle')
                G.add_edge(cycle_node, current_node, 
                           amount=round(current_amt * 0.09, 2), 
                           timestamp=t + 300, 
                           type='layering_return')

            # --- 6. Exit Strategy (The Destination) ---
            # 50% Cuckoo (Innocent), 50% Merchant/Funnel (Guilty)
            if random.random() < 0.5:
                # Cuckoo Smurfing: Exit to innocent existing node
                exit_node = random.choice(existing_nodes)
                # Note: Do NOT label innocent exit node as suspicious
            else:
                # Funnel/Merchant: Exit to new shell account
                exit_node = f"Shell_Exit_{start_time}_{i}"
                G.add_node(exit_node, ntype='merchant', label=1, pattern='shell_exit')
                exit_nodes.append(exit_node)
                
            G.add_edge(current_node, exit_node, 
                       amount=round(peel_amt, 2), 
                       timestamp=t + random.randint(10, 60), 
                       type='peel_exit')

            current_node = next_node

        return {
            'chain_nodes': chain_nodes,
            'wash_nodes': wash_nodes,
            'cycle_nodes': cycle_nodes,
            'exit_nodes': exit_nodes,
            'all_suspicious': chain_nodes + wash_nodes + cycle_nodes + exit_nodes
        }
    
    def inject_simple_peel_chain(
        self,
        G: nx.DiGraph,
        existing_nodes: List[str],
        length: int = 20,
        initial_amount: float = 100000,
        peel_range: Tuple[float, float] = (0.03, 0.08)
    ) -> List[str]:
        """
        Creates a basic peel chain without wash trading camouflage.
        Simpler variant for baseline training.
        
        Args:
            G: NetworkX graph
            existing_nodes: Existing account nodes
            length: Chain length
            initial_amount: Starting amount
            peel_range: Min/max peel percentage per hop
            
        Returns:
            List of chain node IDs
        """
        current_node = random.choice(existing_nodes)
        current_amt = initial_amount
        t = self.base_time
        
        chain_nodes = [current_node]
        
        for i in range(length):
            if current_amt <= 100:
                break
            
            # Time progression
            t += random.randint(3600, 86400)  # 1hr to 24hr
            
            # Peel amount
            peel_pct = random.uniform(*peel_range)
            peel_amt = current_amt * peel_pct
            current_amt -= peel_amt
            
            # Create next hop
            next_node = f"SimplePeel_{i}_{random.randint(1000, 9999)}"
            G.add_node(next_node, ntype='account', label=1, pattern='simple_peel')
            
            # Main chain edge
            G.add_edge(current_node, next_node, 
                       amount=round(current_amt, 2),
                       timestamp=t,
                       type='peel_chain')
            
            # Exit edge for peeled amount
            exit_node = random.choice(existing_nodes)
            G.add_edge(current_node, exit_node,
                       amount=round(peel_amt, 2),
                       timestamp=t + random.randint(60, 300),
                       type='peel_exit')
            
            chain_nodes.append(next_node)
            current_node = next_node
        
        return chain_nodes
    
    def inject_forked_peel_chain(
        self,
        G: nx.DiGraph,
        existing_nodes: List[str],
        initial_length: int = 5,
        fork_count: int = 3,
        branch_length: int = 5,
        initial_amount: float = 100000
    ) -> Dict[str, List[str]]:
        """
        Creates a peel chain that forks into multiple branches.
        Pattern: Main chain splits into N branches after initial hops.
        
        Args:
            G: NetworkX graph
            existing_nodes: Existing nodes
            initial_length: Hops before forking
            fork_count: Number of branches to create
            branch_length: Length of each branch
            initial_amount: Starting amount
            
        Returns:
            Dict with 'main_chain' and 'branches' lists
        """
        t = self.base_time
        current_node = random.choice(existing_nodes)
        current_amt = initial_amount
        
        main_chain = []
        
        # Initial chain before fork
        for i in range(initial_length):
            t += random.randint(600, 3600)
            next_node = f"Fork_Main_{i}_{random.randint(100, 999)}"
            G.add_node(next_node, ntype='account', label=1, pattern='fork_main')
            G.add_edge(current_node, next_node,
                       amount=round(current_amt * 0.95, 2),
                       timestamp=t,
                       type='peel_chain')
            main_chain.append(next_node)
            current_node = next_node
            current_amt *= 0.95
        
        # Fork point - split into branches
        fork_node = current_node
        amount_per_branch = current_amt / fork_count
        branches = []
        
        for b in range(fork_count):
            branch_nodes = []
            branch_amt = amount_per_branch
            branch_node = fork_node
            
            for j in range(branch_length):
                t += random.randint(300, 1800)
                next_node = f"Fork_B{b}_{j}_{random.randint(100, 999)}"
                G.add_node(next_node, ntype='account', label=1, pattern='fork_branch')
                G.add_edge(branch_node, next_node,
                           amount=round(branch_amt * 0.9, 2),
                           timestamp=t,
                           type='peel_chain')
                branch_nodes.append(next_node)
                branch_node = next_node
                branch_amt *= 0.9
            
            branches.append(branch_nodes)
        
        return {
            'main_chain': main_chain,
            'branches': branches,
            'fork_node': fork_node
        }
    
    # ========================================================================
    # STRUCTURING PATTERNS
    # ========================================================================
    
    def inject_smurfing(
        self,
        G: nx.DiGraph,
        target_account: str,
        num_smurfs: int = 20,
        threshold: float = 10000,
        just_below_pct: float = 0.95
    ) -> List[str]:
        """
        Smurfing pattern: Multiple small deposits just below CTR threshold.
        
        Args:
            G: NetworkX graph
            target_account: Account receiving structured deposits
            num_smurfs: Number of smurf accounts
            threshold: CTR threshold (default $10,000)
            just_below_pct: How close to threshold (0.95 = 95%)
            
        Returns:
            List of smurf account IDs
        """
        t = self.base_time
        smurf_accounts = []
        
        for i in range(num_smurfs):
            smurf = f"Smurf_{i}_{random.randint(1000, 9999)}"
            G.add_node(smurf, ntype='account', label=1, pattern='smurfing')
            smurf_accounts.append(smurf)
            
            # Amount just below threshold with some variance
            amount = threshold * random.uniform(just_below_pct - 0.05, just_below_pct)
            
            # Spread over time
            t += random.randint(1800, 7200)  # 30min to 2hr between deposits
            
            G.add_edge(smurf, target_account,
                       amount=round(amount, 2),
                       timestamp=t,
                       type='smurfing_deposit')
        
        return smurf_accounts
    
    def inject_threshold_avoidance(
        self,
        G: nx.DiGraph,
        account: str,
        targets: List[str],
        num_transactions: int = 15,
        threshold: float = 10000
    ) -> List[Tuple[str, str, float]]:
        """
        Account consistently sends amounts 5-15% below reporting threshold.
        
        Args:
            G: NetworkX graph
            account: Source account
            targets: Target accounts
            num_transactions: Number of transactions
            threshold: CTR threshold
            
        Returns:
            List of (source, target, amount) tuples
        """
        t = self.base_time
        transactions = []
        
        for i in range(num_transactions):
            target = random.choice(targets)
            # Amounts consistently 5-15% below threshold
            amount = threshold * random.uniform(0.85, 0.95)
            t += random.randint(3600, 28800)
            
            G.add_edge(account, target,
                       amount=round(amount, 2),
                       timestamp=t,
                       type='threshold_avoidance')
            transactions.append((account, target, amount))
        
        return transactions
    
    # ========================================================================
    # LAYERING PATTERNS
    # ========================================================================
    
    def inject_multi_hop_chain(
        self,
        G: nx.DiGraph,
        start_node: str,
        length: int = 10,
        amount: float = 50000,
        fee_pct: float = 0.02
    ) -> List[str]:
        """
        Simple multi-hop chain: A → B → C → D → E (no return).
        Each hop takes a small "fee" (mule cut).
        
        Args:
            G: NetworkX graph
            start_node: Starting account
            length: Number of hops
            amount: Initial amount
            fee_pct: Percentage taken at each hop
            
        Returns:
            List of chain node IDs
        """
        t = self.base_time
        current = start_node
        current_amt = amount
        chain = [current]
        
        for i in range(length):
            next_node = f"MultiHop_{i}_{random.randint(100, 999)}"
            G.add_node(next_node, ntype='account', label=1, pattern='multi_hop')
            
            t += random.randint(1800, 7200)
            current_amt *= (1 - fee_pct)
            
            G.add_edge(current, next_node,
                       amount=round(current_amt, 2),
                       timestamp=t,
                       type='multi_hop_chain')
            
            chain.append(next_node)
            current = next_node
        
        return chain
    
    def inject_funnel_pattern(
        self,
        G: nx.DiGraph,
        hub_account: str,
        fan_in_sources: List[str],
        fan_out_targets: List[str],
        total_amount: float = 100000
    ) -> Dict[str, List]:
        """
        Funnel pattern: Fan-in → Hub → Fan-out.
        Money concentrates then disperses.
        
        Args:
            G: NetworkX graph
            hub_account: Central hub account
            fan_in_sources: Accounts sending to hub
            fan_out_targets: Accounts receiving from hub
            total_amount: Total amount flowing through
            
        Returns:
            Dict with 'fan_in' and 'fan_out' transaction lists
        """
        t = self.base_time
        amount_per_source = total_amount / len(fan_in_sources)
        
        # Fan-in phase
        fan_in_txs = []
        for src in fan_in_sources:
            t += random.randint(60, 300)
            amt = amount_per_source * random.uniform(0.9, 1.1)
            G.add_edge(src, hub_account,
                       amount=round(amt, 2),
                       timestamp=t,
                       type='funnel_fan_in')
            fan_in_txs.append((src, hub_account, amt))
        
        # Brief pause at hub
        t += random.randint(3600, 7200)
        
        # Fan-out phase
        amount_per_target = (total_amount * 0.95) / len(fan_out_targets)
        fan_out_txs = []
        for tgt in fan_out_targets:
            t += random.randint(60, 300)
            amt = amount_per_target * random.uniform(0.9, 1.1)
            G.add_edge(hub_account, tgt,
                       amount=round(amt, 2),
                       timestamp=t,
                       type='funnel_fan_out')
            fan_out_txs.append((hub_account, tgt, amt))
        
        return {
            'fan_in': fan_in_txs,
            'fan_out': fan_out_txs
        }
    
    # ========================================================================
    # BEHAVIORAL PATTERNS
    # ========================================================================
    
    def inject_dormant_activation(
        self,
        G: nx.DiGraph,
        account: str,
        targets: List[str],
        dormant_days: int = 180,
        burst_transactions: int = 20,
        burst_amount: float = 50000
    ) -> List[Tuple]:
        """
        Dormant account suddenly becomes very active.
        
        Args:
            G: NetworkX graph
            account: The dormant account
            targets: Target accounts for burst activity
            dormant_days: Days of dormancy before burst
            burst_transactions: Number of burst transactions
            burst_amount: Total amount in burst
            
        Returns:
            List of burst transactions
        """
        # Start timestamp (dormant_days ago)
        dormant_start = self.base_time - (dormant_days * 86400)
        
        # Add 1-2 minimal transactions during dormancy
        for i in range(random.randint(1, 2)):
            t = dormant_start + random.randint(0, dormant_days * 86400 // 2)
            G.add_edge(account, random.choice(targets),
                       amount=random.uniform(10, 100),
                       timestamp=t,
                       type='dormant_minimal')
        
        # Burst activity (all within 24-48 hours)
        burst_start = self.base_time - random.randint(86400, 172800)
        transactions = []
        amount_per_tx = burst_amount / burst_transactions
        
        for i in range(burst_transactions):
            t = burst_start + random.randint(0, 86400)
            target = random.choice(targets)
            amt = amount_per_tx * random.uniform(0.8, 1.2)
            
            G.add_edge(account, target,
                       amount=round(amt, 2),
                       timestamp=t,
                       type='dormant_burst')
            transactions.append((account, target, amt, t))
        
        return transactions
    
    def inject_mule_account(
        self,
        G: nx.DiGraph,
        mule: str,
        sources: List[str],
        targets: List[str],
        pass_through_amount: float = 50000
    ) -> Dict:
        """
        Mule account: Receives and quickly sends similar amounts (pass-through).
        
        Args:
            G: NetworkX graph
            mule: The mule account
            sources: Incoming sources
            targets: Outgoing targets
            pass_through_amount: Total amount passing through
            
        Returns:
            Dict with incoming and outgoing transactions
        """
        t = self.base_time
        incoming = []
        outgoing = []
        
        # Receive from multiple sources
        in_per_source = pass_through_amount / len(sources)
        for src in sources:
            amt = in_per_source * random.uniform(0.9, 1.1)
            G.add_edge(src, mule,
                       amount=round(amt, 2),
                       timestamp=t,
                       type='mule_incoming')
            incoming.append((src, mule, amt))
            t += random.randint(60, 600)
        
        # Quick turnaround (within hours)
        t += random.randint(3600, 14400)
        
        # Send to multiple targets (minus small "fee")
        out_per_target = (pass_through_amount * 0.95) / len(targets)
        for tgt in targets:
            amt = out_per_target * random.uniform(0.9, 1.1)
            G.add_edge(mule, tgt,
                       amount=round(amt, 2),
                       timestamp=t,
                       type='mule_outgoing')
            outgoing.append((mule, tgt, amt))
            t += random.randint(60, 600)
        
        return {'incoming': incoming, 'outgoing': outgoing}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_pattern_summary(self, G: nx.DiGraph) -> Dict:
        """
        Get summary of patterns in the graph based on edge types.
        
        Returns:
            Dict mapping pattern type to count
        """
        pattern_counts = {}
        for u, v, data in G.edges(data=True):
            ptype = data.get('type', 'unknown')
            pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
        return pattern_counts
    
    def inject_pattern_from_action(
        self,
        G: nx.DiGraph,
        existing_nodes: List[str],
        action: Dict
    ) -> Dict[str, List[str]]:
        """
        Dispatch to appropriate pattern based on action dict from RL agent.
        
        This method bridges the RL agent's action space to the pattern library.
        Actions are expected to be pre-canonicalized by the environment.
        
        Args:
            G: NetworkX graph to inject pattern into
            existing_nodes: List of existing node IDs
            action: Action dictionary with keys:
                - pattern_type: int [0-4]
                - temporal_mode: int [0-2]
                - chain_length: int [5-20]
                - fork_count: int [1-4]
                - exit_strategy: int [0-2]
                - peel_pct: float [0.02-0.15]
                - wash_intensity: float [0.0-1.0]
                - cycle_probability: float [0.0-0.5]
                
        Returns:
            Dict with 'injected_nodes' list and pattern-specific details
        """
        pattern_type = action.get('pattern_type', 0)
        chain_length = action.get('chain_length', 10)
        fork_count = action.get('fork_count', 2)
        exit_strategy = action.get('exit_strategy', 0)
        peel_pct = action.get('peel_pct', 0.05)
        wash_intensity = action.get('wash_intensity', 0.3)
        cycle_probability = action.get('cycle_probability', 0.2)
        temporal_mode = action.get('temporal_mode', 2)  # mixed default
        
        result = {'injected_nodes': [], 'pattern_type': pattern_type}
        
        # Select initial amount
        initial_amount = random.uniform(50000, 200000)
        
        if pattern_type == 0:  # sophisticated peel chain
            peel_result = self.inject_sophisticated_peel_chain(
                G, existing_nodes,
                length=chain_length,
                initial_amount=initial_amount,
            )
            result['injected_nodes'] = peel_result.get('all_suspicious', [])
            result['details'] = peel_result
            
        elif pattern_type == 1:  # forked peel chain
            fork_result = self.inject_forked_peel_chain(
                G, existing_nodes,
                initial_length=max(3, chain_length // 3),
                fork_count=fork_count,
                branch_length=max(2, chain_length // 2),
                initial_amount=initial_amount,
            )
            main = fork_result.get('main_chain', [])
            branches = fork_result.get('branches', [])
            result['injected_nodes'] = main + [n for b in branches for n in b]
            result['details'] = fork_result
            
        elif pattern_type == 2:  # smurfing
            target = random.choice(existing_nodes)
            smurfs = self.inject_smurfing(
                G, target,
                num_smurfs=chain_length,  # Reinterpret as num_smurfs
            )
            result['injected_nodes'] = smurfs
            result['target'] = target
            
        elif pattern_type == 3:  # funnel
            hub = f"Hub_RL_{int(self.base_time)}_{random.randint(1000, 9999)}"
            G.add_node(hub, ntype='account', label=1, pattern='funnel_hub')
            
            fan_in = random.sample(existing_nodes, k=min(8, len(existing_nodes)))
            fan_out = random.sample(existing_nodes, k=min(5, len(existing_nodes)))
            
            funnel_result = self.inject_funnel_pattern(
                G, hub, fan_in, fan_out,
                total_amount=initial_amount,
            )
            result['injected_nodes'] = [hub]
            result['details'] = funnel_result
            
        elif pattern_type == 4:  # mule
            mule = f"Mule_RL_{int(self.base_time)}_{random.randint(1000, 9999)}"
            G.add_node(mule, ntype='account', label=1, pattern='mule')
            
            sources = random.sample(existing_nodes, k=min(5, len(existing_nodes)))
            targets = random.sample(existing_nodes, k=min(3, len(existing_nodes)))
            
            mule_result = self.inject_mule_account(
                G, mule, sources, targets,
                pass_through_amount=initial_amount * 0.5,
            )
            result['injected_nodes'] = [mule]
            result['details'] = mule_result
        
        return result
    
    def label_nodes_by_pattern(self, G: nx.DiGraph) -> Dict[str, int]:
        """
        Get node labels based on pattern participation.
        
        Returns:
            Dict mapping node ID to label (0=clean, 1=suspicious)
        """
        labels = {}
        for node, data in G.nodes(data=True):
            labels[node] = data.get('label', 0)
        return labels


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_training_graph_with_patterns(
    num_accounts: int = 500,
    num_peel_chains: int = 5,
    num_smurfing_rings: int = 3,
    num_funnels: int = 2,
    num_mules: int = 10,
    seed: int = 42
) -> Tuple[nx.DiGraph, Dict[str, int], PatternLibrary]:
    """
    Generate a complete training graph with multiple pattern types.
    
    Args:
        num_accounts: Number of base accounts
        num_peel_chains: Number of peel chains to inject
        num_smurfing_rings: Number of smurfing operations
        num_funnels: Number of funnel patterns
        num_mules: Number of mule accounts
        seed: Random seed
        
    Returns:
        Tuple of (graph, node_labels, pattern_library)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.DiGraph()
    
    # Create base accounts
    accounts = [f"ACC_{i:05d}" for i in range(num_accounts)]
    for acc in accounts:
        G.add_node(acc, ntype='account', label=0, pattern='organic')
    
    # Add organic background traffic
    for _ in range(num_accounts * 3):
        src = random.choice(accounts)
        dst = random.choice(accounts)
        if src != dst:
            G.add_edge(src, dst,
                       amount=random.uniform(10, 1000),
                       timestamp=int(time.time()) - random.randint(0, 30*86400),
                       type='organic')
    
    # Initialize pattern library
    lib = PatternLibrary(seed=seed)
    
    # Inject peel chains
    for _ in range(num_peel_chains):
        lib.inject_sophisticated_peel_chain(
            G, accounts, 
            length=random.randint(10, 20),
            initial_amount=random.uniform(50000, 200000)
        )
    
    # Inject smurfing
    for _ in range(num_smurfing_rings):
        target = random.choice(accounts)
        lib.inject_smurfing(G, target, num_smurfs=random.randint(10, 25))
    
    # Inject funnels
    for _ in range(num_funnels):
        hub = f"Hub_{random.randint(1000, 9999)}"
        G.add_node(hub, ntype='account', label=1, pattern='funnel_hub')
        fan_in = random.sample(accounts, k=random.randint(5, 10))
        fan_out = random.sample(accounts, k=random.randint(3, 7))
        lib.inject_funnel_pattern(G, hub, fan_in, fan_out)
    
    # Inject mules
    mule_accounts = []
    for i in range(num_mules):
        mule = f"Mule_{i}_{random.randint(100, 999)}"
        G.add_node(mule, ntype='account', label=1, pattern='mule')
        mule_accounts.append(mule)
        sources = random.sample(accounts, k=random.randint(3, 8))
        targets = random.sample(accounts, k=random.randint(2, 5))
        lib.inject_mule_account(G, mule, sources, targets)
    
    # Extract labels
    labels = lib.label_nodes_by_pattern(G)
    
    return G, labels, lib


if __name__ == "__main__":
    print("Generating training graph with AML patterns...")
    G, labels, lib = generate_training_graph_with_patterns(
        num_accounts=300,
        num_peel_chains=5,
        num_smurfing_rings=3,
        num_funnels=2,
        num_mules=8,
        seed=42
    )
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Suspicious nodes: {sum(1 for l in labels.values() if l == 1)}")
    
    print(f"\nPattern Summary:")
    for ptype, count in lib.get_pattern_summary(G).items():
        print(f"  {ptype}: {count}")
