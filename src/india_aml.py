"""
India-specific AML feature detectors.

Covers:
  1. UPI transaction code validation & suspicious-pattern detection
  2. NACH mandate abuse patterns
  3. Shell company CIN graph analysis
  4. Benami indicator signals
"""

import re
import random
import string
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx


# ---------------------------------------------------------------------------
# 1. UPI Transaction Codes
# ---------------------------------------------------------------------------

# UPI VPA (Virtual Payment Address) pattern: localpart@bankhandle
_UPI_VPA_RE = re.compile(
    r'^[a-zA-Z0-9._\-]{2,256}@[a-zA-Z]{2,64}$'
)

# UPI Transaction Reference Number (UTR): exactly 12 digits
_UPI_UTR_RE = re.compile(r'^\d{12}$')

# Known UPI PSP handles recognised by NPCI
KNOWN_UPI_HANDLES = {
    "paytm", "upi", "ybl", "okaxis", "okhdfcbank", "okicici", "oksbi",
    "ibl", "axl", "kotak", "federal", "rbl", "pnb", "sbi", "icici",
    "hdfcbank", "aubank", "indus", "barodampay", "pthdfc", "ptaxis",
    "timecosmos", "jupiteraxis", "apl", "digikhata", "abfspay",
}


def validate_upi_vpa(vpa: str) -> dict:
    """
    Validate a UPI VPA and return metadata.

    Returns a dict with keys:
      valid (bool), handle (str|None), known_handle (bool), flags (list[str])
    """
    flags = []
    if not _UPI_VPA_RE.match(vpa):
        return {"valid": False, "handle": None, "known_handle": False, "flags": ["malformed_vpa"]}

    local, handle = vpa.rsplit("@", 1)
    known = handle.lower() in KNOWN_UPI_HANDLES

    if not known:
        flags.append("unknown_psp_handle")
    if len(local) < 3:
        flags.append("suspiciously_short_localpart")
    if re.fullmatch(r'\d+', local):
        flags.append("numeric_only_localpart")  # often auto-generated mule accounts

    return {"valid": True, "handle": handle.lower(), "known_handle": known, "flags": flags}


def validate_utr(utr: str) -> bool:
    """Return True if utr is a syntactically valid UPI Transaction Reference."""
    return bool(_UPI_UTR_RE.match(str(utr).strip()))


class UPIPatternDetector:
    """
    Analyses a list of UPI transaction records for AML-relevant patterns.

    Each record must be a dict with keys:
      utr (str), vpa_src (str), vpa_dst (str), amount (float),
      timestamp (int/epoch-seconds), ref_id (str, optional)
    """

    # Transactions below this threshold qualify as structuring candidates (INR)
    STRUCTURING_THRESHOLD = 200_000  # ₹2 lakh – RBI reporting threshold

    def __init__(self, transactions: list[dict]):
        self.txns = transactions

    def detect_structuring(self, window_seconds: int = 86400, max_amount: float = STRUCTURING_THRESHOLD) -> list[dict]:
        """
        Flag accounts that split a large sum across many sub-threshold UPI
        transactions within a rolling time window (smurfing/structuring).
        """
        by_src: dict[str, list] = defaultdict(list)
        for t in self.txns:
            if t["amount"] < max_amount:
                by_src[t["vpa_src"]].append(t)

        alerts = []
        for src, txs in by_src.items():
            txs_sorted = sorted(txs, key=lambda x: x["timestamp"])
            for i, tx in enumerate(txs_sorted):
                window_txs = [
                    t for t in txs_sorted[i:]
                    if t["timestamp"] - tx["timestamp"] <= window_seconds
                ]
                total = sum(t["amount"] for t in window_txs)
                if len(window_txs) >= 5 and total >= max_amount:
                    alerts.append({
                        "signal": "UPI_STRUCTURING",
                        "vpa_src": src,
                        "num_txns": len(window_txs),
                        "total_amount": round(total, 2),
                        "window_start": tx["timestamp"],
                    })
                    break  # one alert per source per scan

        return alerts

    def detect_rapid_roundtrip(self, window_seconds: int = 3600) -> list[dict]:
        """
        Flag (A→B, B→A) pairs where both legs happen within window_seconds.
        Rapid round-trips are a classic layering signal in UPI networks.
        """
        # Index by (src, dst)
        forward: dict[tuple, list] = defaultdict(list)
        for t in self.txns:
            forward[(t["vpa_src"], t["vpa_dst"])].append(t)

        alerts = []
        checked = set()
        for (src, dst), fwd_txs in forward.items():
            if (dst, src) not in forward:
                continue
            pair_key = tuple(sorted([src, dst]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            rev_txs = forward[(dst, src)]
            for ft in fwd_txs:
                for rt in rev_txs:
                    if abs(rt["timestamp"] - ft["timestamp"]) <= window_seconds:
                        alerts.append({
                            "signal": "UPI_RAPID_ROUNDTRIP",
                            "leg1": {"src": src, "dst": dst, "amount": ft["amount"], "utr": ft["utr"]},
                            "leg2": {"src": dst, "dst": src, "amount": rt["amount"], "utr": rt["utr"]},
                            "gap_seconds": abs(rt["timestamp"] - ft["timestamp"]),
                        })
                        break
        return alerts

    def detect_mule_vpa(self) -> list[dict]:
        """
        Flag VPAs that exhibit mule-account characteristics:
          - Numeric-only local part (auto-generated)
          - Unknown PSP handle
          - Exclusively receive then immediately forward (no organic spending)
        """
        receive_ts: dict[str, list] = defaultdict(list)
        send_ts: dict[str, list] = defaultdict(list)
        for t in self.txns:
            receive_ts[t["vpa_dst"]].append(t["timestamp"])
            send_ts[t["vpa_src"]].append(t["timestamp"])

        alerts = []
        all_vpas = set(receive_ts) | set(send_ts)
        for vpa in all_vpas:
            meta = validate_upi_vpa(vpa)
            flags = list(meta.get("flags", []))

            # Pass-through mule: receives and forwards within tight window
            rcv = sorted(receive_ts.get(vpa, []))
            snd = sorted(send_ts.get(vpa, []))
            if rcv and snd:
                pass_through_count = sum(
                    1 for r in rcv
                    if any(0 < s - r <= 1800 for s in snd)  # forwarded within 30 min
                )
                if pass_through_count >= 3:
                    flags.append("pass_through_mule")

            if flags:
                alerts.append({"signal": "UPI_MULE_VPA", "vpa": vpa, "flags": flags})

        return alerts


# ---------------------------------------------------------------------------
# 2. NACH Mandate Patterns
# ---------------------------------------------------------------------------

# NACH Mandate Reference format: NACH + 4-char bank code + 12 digits
_NACH_REF_RE = re.compile(r'^NACH[A-Z0-9]{4}\d{12}$')

# NPCI-registered mandate categories
NACH_CATEGORIES = {"LOAN_EMI", "INSURANCE", "UTILITY", "INVESTMENT", "SUBSCRIPTION", "OTHER"}


def validate_nach_ref(ref: str) -> bool:
    return bool(_NACH_REF_RE.match(str(ref).strip().upper()))


class NACHMandateAnalyzer:
    """
    Detects NACH mandate abuse patterns.

    Each mandate record:
      mandate_ref (str), debtor_account (str), creditor_account (str),
      amount (float), frequency (str: DAILY/WEEKLY/MONTHLY/ADHOC),
      category (str), start_date (int/epoch), end_date (int/epoch|None),
      status (str: ACTIVE/CANCELLED/FAILED)

    Each debit record:
      mandate_ref (str), amount (float), timestamp (int/epoch),
      status (str: SUCCESS/FAILED/RETURNED)
    """

    def __init__(self, mandates: list[dict], debits: list[dict]):
        self.mandates = {m["mandate_ref"]: m for m in mandates}
        self.debits = debits
        self._debits_by_mandate: dict[str, list] = defaultdict(list)
        for d in debits:
            self._debits_by_mandate[d["mandate_ref"]].append(d)

    def detect_amount_escalation(self, escalation_factor: float = 2.0) -> list[dict]:
        """
        Flag mandates where debit amounts consistently exceed the registered
        mandate amount (possible unauthorised debit or fraud).
        """
        alerts = []
        for ref, m in self.mandates.items():
            debits = self._debits_by_mandate.get(ref, [])
            overages = [d for d in debits if d["amount"] > m["amount"] * escalation_factor]
            if overages:
                alerts.append({
                    "signal": "NACH_AMOUNT_ESCALATION",
                    "mandate_ref": ref,
                    "registered_amount": m["amount"],
                    "overage_count": len(overages),
                    "max_debit": max(d["amount"] for d in overages),
                })
        return alerts

    def detect_mandate_churn(self, churn_window_days: int = 30, churn_threshold: int = 3) -> list[dict]:
        """
        Flag debtors who create and cancel many mandates rapidly – a pattern
        seen in account takeover and money-mule recruitment.
        """
        by_debtor: dict[str, list] = defaultdict(list)
        for m in self.mandates.values():
            by_debtor[m["debtor_account"]].append(m)

        alerts = []
        window_secs = churn_window_days * 86400
        for debtor, mands in by_debtor.items():
            cancelled = [m for m in mands if m["status"] == "CANCELLED"]
            cancelled_sorted = sorted(cancelled, key=lambda x: x["start_date"])
            for i, m in enumerate(cancelled_sorted):
                window_group = [
                    c for c in cancelled_sorted[i:]
                    if c["start_date"] - m["start_date"] <= window_secs
                ]
                if len(window_group) >= churn_threshold:
                    alerts.append({
                        "signal": "NACH_MANDATE_CHURN",
                        "debtor_account": debtor,
                        "cancelled_in_window": len(window_group),
                        "window_days": churn_window_days,
                    })
                    break
        return alerts

    def detect_high_failure_rate(self, min_attempts: int = 5, failure_rate_threshold: float = 0.6) -> list[dict]:
        """
        Mandates with consistently high failure rates can indicate accounts
        used to probe available balances (account balance enumeration).
        """
        alerts = []
        for ref, debits in self._debits_by_mandate.items():
            if len(debits) < min_attempts:
                continue
            failed = [d for d in debits if d["status"] in ("FAILED", "RETURNED")]
            rate = len(failed) / len(debits)
            if rate >= failure_rate_threshold:
                alerts.append({
                    "signal": "NACH_HIGH_FAILURE_RATE",
                    "mandate_ref": ref,
                    "total_attempts": len(debits),
                    "failure_rate": round(rate, 3),
                })
        return alerts

    def detect_ghost_mandate(self, days_without_debit: int = 180) -> list[dict]:
        """
        Active mandates with zero debits for an extended period followed by
        a sudden large debit – a dormancy-then-drain pattern.
        """
        alerts = []
        now = int(datetime.utcnow().timestamp())
        for ref, m in self.mandates.items():
            if m["status"] != "ACTIVE":
                continue
            debits = sorted(self._debits_by_mandate.get(ref, []), key=lambda x: x["timestamp"])
            if not debits:
                continue
            # Gap between consecutive debits
            for i in range(1, len(debits)):
                gap = debits[i]["timestamp"] - debits[i - 1]["timestamp"]
                if gap > days_without_debit * 86400:
                    alerts.append({
                        "signal": "NACH_GHOST_MANDATE",
                        "mandate_ref": ref,
                        "gap_days": round(gap / 86400),
                        "sudden_debit_amount": debits[i]["amount"],
                    })
                    break
        return alerts


# ---------------------------------------------------------------------------
# 3. Shell Company CIN Graph
# ---------------------------------------------------------------------------

# CIN format: (L|U) + 5-digit NIC code + 2-letter state + 4-digit year + (PLC|OPC|LLC|NPL|...) + 6-digit reg no
_CIN_RE = re.compile(
    r'^[LU]\d{5}[A-Z]{2}\d{4}(PLC|OPC|LLC|NPL|FLC|GOI|SGC|GAP)\d{6}$'
)

INDIAN_STATES = {
    "AN","AP","AR","AS","BR","CG","CH","DH","DL","DN","GA","GJ","HP",
    "HR","JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL",
    "OR","PB","PY","RJ","SK","TG","TN","TR","UK","UP","WB",
}


def validate_cin(cin: str) -> dict:
    """Parse and validate a Corporate Identification Number (CIN)."""
    cin = cin.strip().upper()
    if not _CIN_RE.match(cin):
        return {"valid": False, "raw": cin}
    return {
        "valid": True,
        "raw": cin,
        "listing_status": "Listed" if cin[0] == "L" else "Unlisted",
        "nic_code": cin[1:6],
        "state": cin[6:8],
        "year": int(cin[8:12]),
        "company_type": re.search(r'(PLC|OPC|LLC|NPL|FLC|GOI|SGC|GAP)', cin).group(),
        "reg_number": cin[-6:],
    }


class ShellCompanyCINGraph:
    """
    Builds and analyses a director–company bipartite graph to surface
    shell-company networks using CIN and director DIN metadata.

    Node types:
      company  – identified by CIN
      director – identified by DIN (8-digit Director Identification Number)

    Each company record:
      cin (str), directors (list[str] of DINs), paid_up_capital (float),
      registered_year (int), registered_state (str),
      annual_transactions (float|None)

    Edges: director <-> company
    """

    # A company with very low paid-up capital but high transaction volume
    CAPITAL_TO_TXNS_RATIO_THRESHOLD = 50.0
    # Companies registered within this many years are "fresh" – higher risk
    FRESH_COMPANY_YEARS = 3

    def __init__(self, companies: list[dict]):
        self.companies = {c["cin"]: c for c in companies}
        self.G = self._build_graph(companies)

    def _build_graph(self, companies: list[dict]) -> nx.Graph:
        G = nx.Graph()
        for c in companies:
            cin_meta = validate_cin(c["cin"])
            G.add_node(c["cin"], node_type="company", **cin_meta, **c)
            for din in c.get("directors", []):
                G.add_node(din, node_type="director")
                G.add_edge(din, c["cin"])
        return G

    def detect_circular_directorships(self, min_shared_directors: int = 2) -> list[dict]:
        """
        Find pairs of companies that share >= min_shared_directors directors.
        A dense cross-company director network is a strong shell-company signal.
        """
        cin_to_directors: dict[str, set] = {}
        for cin, c in self.companies.items():
            cin_to_directors[cin] = set(c.get("directors", []))

        cins = list(cin_to_directors)
        alerts = []
        for i in range(len(cins)):
            for j in range(i + 1, len(cins)):
                shared = cin_to_directors[cins[i]] & cin_to_directors[cins[j]]
                if len(shared) >= min_shared_directors:
                    alerts.append({
                        "signal": "CIN_CIRCULAR_DIRECTORSHIP",
                        "company_a": cins[i],
                        "company_b": cins[j],
                        "shared_directors": list(shared),
                        "shared_count": len(shared),
                    })
        return alerts

    def detect_director_hubs(self, max_directorships: int = 10) -> list[dict]:
        """
        Flag directors sitting on an unusually large number of boards.
        Indian Companies Act limits to 20 (public) / 10 (listed), but
        lower thresholds indicate network controllers.
        """
        alerts = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") != "director":
                continue
            degree = self.G.degree(node)
            if degree > max_directorships:
                companies = list(self.G.neighbors(node))
                alerts.append({
                    "signal": "CIN_DIRECTOR_HUB",
                    "din": node,
                    "num_companies": degree,
                    "companies": companies,
                })
        return alerts

    def detect_fresh_shell_candidates(self, current_year: int | None = None) -> list[dict]:
        """
        Flag recently incorporated companies with low paid-up capital but
        high transaction volumes – classic placement-layer shell profile.
        """
        if current_year is None:
            current_year = datetime.utcnow().year
        alerts = []
        for cin, c in self.companies.items():
            age = current_year - c.get("registered_year", current_year)
            capital = c.get("paid_up_capital", 0)
            txns = c.get("annual_transactions", 0) or 0
            flags = []
            if age <= self.FRESH_COMPANY_YEARS:
                flags.append("recently_incorporated")
            if capital > 0 and txns / capital > self.CAPITAL_TO_TXNS_RATIO_THRESHOLD:
                flags.append("high_txn_to_capital_ratio")
            if len(c.get("directors", [])) <= 2:
                flags.append("minimal_directors")
            if flags:
                alerts.append({
                    "signal": "CIN_SHELL_CANDIDATE",
                    "cin": cin,
                    "age_years": age,
                    "paid_up_capital": capital,
                    "annual_transactions": txns,
                    "flags": flags,
                })
        return alerts

    def get_connected_components(self) -> list[list[str]]:
        """Return clusters of interconnected CINs (via shared directors)."""
        company_nodes = {n for n, d in self.G.nodes(data=True) if d.get("node_type") == "company"}
        clusters = []
        for component in nx.connected_components(self.G):
            cin_nodes = [n for n in component if n in company_nodes]
            if len(cin_nodes) >= 2:
                clusters.append(sorted(cin_nodes))
        return sorted(clusters, key=len, reverse=True)


# ---------------------------------------------------------------------------
# 4. Benami Indicator Signals
# ---------------------------------------------------------------------------

class BenamiIndicatorDetector:
    """
    Detects signals consistent with benami transactions under the
    Prohibition of Benami Property Transactions Act, 1988 (amended 2016).

    A benami transaction occurs when property/assets are held by one person
    (benamidar) but paid for by another (beneficial owner), often to conceal
    the true ownership.

    Transaction record keys:
      txn_id (str), payer_id (str), beneficiary_id (str),
      amount (float), timestamp (int/epoch),
      declared_purpose (str), asset_type (str: PROPERTY/EQUITY/CASH/OTHER),
      pan_payer (str|None), pan_beneficiary (str|None),
      payer_declared_income (float|None),   # annual
      relationship (str|None: FAMILY/ASSOCIATE/UNKNOWN)
    """

    # PAN format: 5 letters + 4 digits + 1 letter
    _PAN_RE = re.compile(r'^[A-Z]{5}\d{4}[A-Z]$')

    # Amount multiple of declared annual income that triggers scrutiny
    INCOME_MULTIPLE_THRESHOLD = 3.0

    def __init__(self, transactions: list[dict]):
        self.txns = transactions

    def validate_pan(self, pan: str) -> bool:
        return bool(self._PAN_RE.match(str(pan).strip().upper()))

    def detect_income_mismatch(self) -> list[dict]:
        """
        Flag transactions where the amount significantly exceeds the payer's
        declared annual income – a core benami red flag.
        """
        alerts = []
        for t in self.txns:
            income = t.get("payer_declared_income")
            if not income or income <= 0:
                continue
            if t["amount"] > income * self.INCOME_MULTIPLE_THRESHOLD:
                alerts.append({
                    "signal": "BENAMI_INCOME_MISMATCH",
                    "txn_id": t["txn_id"],
                    "payer_id": t["payer_id"],
                    "amount": t["amount"],
                    "declared_income": income,
                    "multiple": round(t["amount"] / income, 2),
                })
        return alerts

    def detect_third_party_property(self) -> list[dict]:
        """
        Flag property/high-value asset transactions where the beneficiary
        has no declared financial relationship with the payer (unknown/unrelated).
        """
        alerts = []
        for t in self.txns:
            if t.get("asset_type") not in ("PROPERTY", "EQUITY"):
                continue
            if t.get("relationship", "UNKNOWN") == "UNKNOWN" and t["amount"] >= 500_000:
                flags = ["unrelated_third_party_beneficiary"]
                pan_b = t.get("pan_beneficiary")
                if pan_b and not self.validate_pan(pan_b):
                    flags.append("invalid_pan_beneficiary")
                alerts.append({
                    "signal": "BENAMI_THIRD_PARTY_PROPERTY",
                    "txn_id": t["txn_id"],
                    "payer_id": t["payer_id"],
                    "beneficiary_id": t["beneficiary_id"],
                    "amount": t["amount"],
                    "asset_type": t["asset_type"],
                    "flags": flags,
                })
        return alerts

    def detect_round_trip_cash(self, window_seconds: int = 7 * 86400) -> list[dict]:
        """
        Detect cash/transfer round-trips: A pays B, B returns equivalent
        amount to A within the window. Used to create paper trails for
        benami asset purchases while actual funds stay with A.
        """
        by_pair: dict[tuple, list] = defaultdict(list)
        for t in self.txns:
            by_pair[(t["payer_id"], t["beneficiary_id"])].append(t)

        alerts = []
        checked = set()
        for (payer, ben), fwd_txs in by_pair.items():
            rev_key = (ben, payer)
            if rev_key not in by_pair:
                continue
            pair_key = tuple(sorted([payer, ben]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            rev_txs = by_pair[rev_key]
            for ft in fwd_txs:
                for rt in rev_txs:
                    amount_close = abs(rt["amount"] - ft["amount"]) / max(ft["amount"], 1) < 0.05
                    time_ok = 0 < rt["timestamp"] - ft["timestamp"] <= window_seconds
                    if amount_close and time_ok:
                        alerts.append({
                            "signal": "BENAMI_ROUND_TRIP_CASH",
                            "leg1_txn_id": ft["txn_id"],
                            "leg2_txn_id": rt["txn_id"],
                            "payer_id": payer,
                            "beneficiary_id": ben,
                            "amount": ft["amount"],
                            "return_gap_days": round((rt["timestamp"] - ft["timestamp"]) / 86400, 1),
                        })
                        break
        return alerts

    def detect_pan_mismatch_cluster(self) -> list[dict]:
        """
        Flag payer_ids that use multiple different PANs across transactions
        (identity fragmentation) – common in benami networks where individuals
        operate through multiple identities.
        """
        payer_pans: dict[str, set] = defaultdict(set)
        for t in self.txns:
            pan = t.get("pan_payer")
            if pan and self.validate_pan(pan):
                payer_pans[t["payer_id"]].add(pan)

        alerts = []
        for payer_id, pans in payer_pans.items():
            if len(pans) > 1:
                alerts.append({
                    "signal": "BENAMI_PAN_FRAGMENTATION",
                    "payer_id": payer_id,
                    "num_pans": len(pans),
                    "pans": list(pans),
                })
        return alerts


# ---------------------------------------------------------------------------
# Simulator helpers – generate synthetic India-specific test data
# ---------------------------------------------------------------------------

def _random_utr() -> str:
    return "".join(random.choices(string.digits, k=12))


def _random_vpa(handles=None, mule=False) -> str:
    if handles is None:
        handles = list(KNOWN_UPI_HANDLES)
    if mule:
        local = "".join(random.choices(string.digits, k=10))
        handle = random.choice(["newpsp", "fastpay", "quickupi"])
    else:
        local = "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
        handle = random.choice(handles)
    return f"{local}@{handle}"


def _random_cin(year_range=(2010, 2024)) -> str:
    status = random.choice(["L", "U"])
    nic = f"{random.randint(10000, 99999)}"
    state = random.choice(list(INDIAN_STATES))
    year = random.randint(*year_range)
    ctype = random.choice(["PLC", "OPC"])
    reg = f"{random.randint(100000, 999999)}"
    return f"{status}{nic}{state}{year}{ctype}{reg}"


def _random_din() -> str:
    return f"{random.randint(10000000, 99999999)}"


def _random_pan() -> str:
    letters = string.ascii_uppercase
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices(string.digits, k=4))
        + random.choice(letters)
    )


def generate_upi_transactions(
    num_organic: int = 200,
    num_structuring_actors: int = 3,
    num_roundtrip_pairs: int = 2,
    num_mule_accounts: int = 5,
    base_time: int | None = None,
) -> list[dict]:
    """
    Generate a synthetic list of UPI transaction dicts for testing
    UPIPatternDetector.
    """
    if base_time is None:
        base_time = int(datetime.utcnow().timestamp()) - 7 * 86400

    txns = []

    # Organic traffic
    vpas = [_random_vpa() for _ in range(50)]
    for _ in range(num_organic):
        src, dst = random.sample(vpas, 2)
        txns.append({
            "utr": _random_utr(),
            "vpa_src": src,
            "vpa_dst": dst,
            "amount": round(random.uniform(100, 50_000), 2),
            "timestamp": base_time + random.randint(0, 7 * 86400),
        })

    # Structuring pattern: many small txns just below ₹2 lakh in one day
    for _ in range(num_structuring_actors):
        actor_vpa = _random_vpa()
        burst_time = base_time + random.randint(0, 6 * 86400)
        for _ in range(random.randint(6, 12)):
            dst = random.choice(vpas)
            txns.append({
                "utr": _random_utr(),
                "vpa_src": actor_vpa,
                "vpa_dst": dst,
                "amount": round(random.uniform(15_000, 19_900), 2),
                "timestamp": burst_time + random.randint(0, 3600),
            })

    # Rapid round-trip pairs
    for _ in range(num_roundtrip_pairs):
        vpa_a = _random_vpa()
        vpa_b = _random_vpa()
        t0 = base_time + random.randint(0, 6 * 86400)
        amount = round(random.uniform(500_000, 2_000_000), 2)
        txns.append({"utr": _random_utr(), "vpa_src": vpa_a, "vpa_dst": vpa_b,
                      "amount": amount, "timestamp": t0})
        txns.append({"utr": _random_utr(), "vpa_src": vpa_b, "vpa_dst": vpa_a,
                      "amount": round(amount * random.uniform(0.97, 1.0), 2),
                      "timestamp": t0 + random.randint(60, 1800)})

    # Mule VPAs (numeric local part, unknown handle)
    mule_vpas = [_random_vpa(mule=True) for _ in range(num_mule_accounts)]
    for mule in mule_vpas:
        # Receive from multiple sources then forward quickly
        for _ in range(random.randint(3, 6)):
            src = random.choice(vpas)
            t0 = base_time + random.randint(0, 6 * 86400)
            amount = round(random.uniform(5_000, 50_000), 2)
            txns.append({"utr": _random_utr(), "vpa_src": src, "vpa_dst": mule,
                          "amount": amount, "timestamp": t0})
            txns.append({"utr": _random_utr(), "vpa_src": mule, "vpa_dst": random.choice(vpas),
                          "amount": round(amount * 0.98, 2), "timestamp": t0 + random.randint(300, 1200)})

    return txns


def generate_nach_data(
    num_mandates: int = 30,
    num_debits_per_mandate: int = 12,
    base_date: datetime | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (mandates, debits) lists for testing NACHMandateAnalyzer."""
    if base_date is None:
        base_date = datetime.utcnow() - timedelta(days=365)

    accounts = [f"ACC_{i:05d}" for i in range(50)]
    mandates, debits = [], []

    for i in range(num_mandates):
        ref = f"NACH{random.choice(['HDFC','ICIC','SBIN','UTIB'])}{random.randint(100000000000, 999999999999)}"
        debtor = random.choice(accounts)
        creditor = random.choice(accounts)
        amount = round(random.uniform(1_000, 50_000), 2)
        start_ts = int((base_date + timedelta(days=random.randint(0, 300))).timestamp())
        status = random.choices(["ACTIVE", "CANCELLED"], weights=[0.75, 0.25])[0]

        mandates.append({
            "mandate_ref": ref,
            "debtor_account": debtor,
            "creditor_account": creditor,
            "amount": amount,
            "frequency": random.choice(["MONTHLY", "WEEKLY", "DAILY", "ADHOC"]),
            "category": random.choice(list(NACH_CATEGORIES)),
            "start_date": start_ts,
            "end_date": None,
            "status": status,
        })

        # Generate debits
        for j in range(num_debits_per_mandate):
            d_ts = start_ts + j * random.randint(25 * 86400, 35 * 86400)
            # Occasionally escalate amount or fail
            d_amount = amount * random.choice([1.0, 1.0, 1.0, 3.5])  # 25% chance escalation
            d_status = random.choices(
                ["SUCCESS", "FAILED", "RETURNED"], weights=[0.7, 0.2, 0.1]
            )[0]
            debits.append({
                "mandate_ref": ref,
                "amount": round(d_amount, 2),
                "timestamp": d_ts,
                "status": d_status,
            })

    return mandates, debits


def generate_cin_data(
    num_companies: int = 20,
    num_shell_clusters: int = 2,
    cluster_size: int = 4,
) -> list[dict]:
    """
    Generate synthetic company records for testing ShellCompanyCINGraph.
    Includes deliberately overlapping director pools for shell clusters.
    """
    companies = []
    current_year = datetime.utcnow().year

    # Organic companies
    for _ in range(num_companies):
        cin = _random_cin(year_range=(2000, current_year - 5))
        directors = [_random_din() for _ in range(random.randint(3, 8))]
        companies.append({
            "cin": cin,
            "directors": directors,
            "paid_up_capital": round(random.uniform(100_000, 10_000_000), 2),
            "registered_year": int(cin[8:12]),
            "registered_state": cin[6:8],
            "annual_transactions": round(random.uniform(50_000, 2_000_000), 2),
        })

    # Shell clusters: small set of shared directors control many fresh companies
    for _ in range(num_shell_clusters):
        shared_dins = [_random_din() for _ in range(2)]  # 2 shared directors
        for _ in range(cluster_size):
            cin = _random_cin(year_range=(current_year - 2, current_year))
            extra_din = _random_din()
            companies.append({
                "cin": cin,
                "directors": shared_dins + [extra_din],
                "paid_up_capital": round(random.uniform(100_000, 500_000), 2),
                "registered_year": int(cin[8:12]),
                "registered_state": cin[6:8],
                "annual_transactions": round(random.uniform(50_000_000, 200_000_000), 2),
            })

    return companies


def generate_benami_transactions(
    num_organic: int = 100,
    num_benami_actors: int = 5,
    base_time: int | None = None,
) -> list[dict]:
    """Generate synthetic transaction records for testing BenamiIndicatorDetector."""
    if base_time is None:
        base_time = int(datetime.utcnow().timestamp()) - 30 * 86400

    individuals = [f"IND_{i:04d}" for i in range(40)]
    txns = []
    txn_counter = [0]

    def _txn(**kwargs) -> dict:
        txn_counter[0] += 1
        return {"txn_id": f"TXN_{txn_counter[0]:06d}", **kwargs}

    # Organic
    for _ in range(num_organic):
        payer, ben = random.sample(individuals, 2)
        income = round(random.uniform(300_000, 2_000_000), 2)
        txns.append(_txn(
            payer_id=payer, beneficiary_id=ben,
            amount=round(random.uniform(1_000, income * 0.5), 2),
            timestamp=base_time + random.randint(0, 30 * 86400),
            declared_purpose="PERSONAL_TRANSFER",
            asset_type="CASH",
            pan_payer=_random_pan(), pan_beneficiary=_random_pan(),
            payer_declared_income=income,
            relationship=random.choice(["FAMILY", "ASSOCIATE"]),
        ))

    # Benami actors: pay large amounts to unrelated parties for property
    for _ in range(num_benami_actors):
        payer = random.choice(individuals)
        ben = random.choice([i for i in individuals if i != payer])
        income = round(random.uniform(200_000, 500_000), 2)  # low income
        property_amount = round(random.uniform(2_000_000, 10_000_000), 2)
        t0 = base_time + random.randint(0, 25 * 86400)

        # Forward leg: payer sends large sum to benamidar
        txns.append(_txn(
            payer_id=payer, beneficiary_id=ben,
            amount=property_amount,
            timestamp=t0,
            declared_purpose="PROPERTY_PURCHASE",
            asset_type="PROPERTY",
            pan_payer=_random_pan(), pan_beneficiary=_random_pan(),
            payer_declared_income=income,
            relationship="UNKNOWN",
        ))

        # Return leg within a week (round-trip)
        txns.append(_txn(
            payer_id=ben, beneficiary_id=payer,
            amount=round(property_amount * random.uniform(0.97, 1.0), 2),
            timestamp=t0 + random.randint(86400, 5 * 86400),
            declared_purpose="LOAN_REPAYMENT",
            asset_type="CASH",
            pan_payer=_random_pan(), pan_beneficiary=_random_pan(),
            payer_declared_income=round(random.uniform(500_000, 1_000_000), 2),
            relationship="UNKNOWN",
        ))

        # PAN fragmentation: same payer uses multiple PANs
        for _ in range(2):
            txns.append(_txn(
                payer_id=payer, beneficiary_id=random.choice(individuals),
                amount=round(random.uniform(50_000, 500_000), 2),
                timestamp=base_time + random.randint(0, 30 * 86400),
                declared_purpose="PERSONAL_TRANSFER",
                asset_type="CASH",
                pan_payer=_random_pan(),  # different PAN each time
                pan_beneficiary=_random_pan(),
                payer_declared_income=income,
                relationship="UNKNOWN",
            ))

    return txns
