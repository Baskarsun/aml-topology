"""
Account Detail Page
====================
New page providing a full per-account investigation view.

Features:
  ✓ Account selector from simulation results with risk tier and alert-state badges
  ✓ Overview table of all accounts when none selected
  ✓ Risk score gauge (colour-coded with score bar)
  ✓ Active-signals breakdown with per-phase attribution
  ✓ Alert management: status dropdown + investigation notes (persisted via alert_store)
  ✓ Transaction timeline chart (incoming vs outgoing, filterable by date range)
  ✓ Counterparty mini-graph (interactive Plotly, colour-coded by tier)
  ✓ All charts use plotly_dark theme for visual consistency
"""
import os
import sys
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alert_store import get_alert_state, get_all_states, set_alert_state

TIER_COLORS = {
    "CRITICAL": "#FF2B2B",
    "HIGH":     "#FF6B35",
    "MEDIUM":   "#FFB800",
    "LOW":      "#00FF94",
    "CLEAN":    "#9090a0",
}
TIER_SYMBOLS = {"CRITICAL": "▲", "HIGH": "▲", "MEDIUM": "◆", "LOW": "✓", "CLEAN": "○"}
STATE_ICONS  = {
    "Unreviewed":   "",
    "Investigating": "🔵 ",
    "False Positive": "✅ ",
    "Escalated":    "🔴 ",
}
STATES       = ["Unreviewed", "Investigating", "False Positive", "Escalated"]

# Mapping from signal keyword → (icon, phase name, colour, human description)
_PHASE_SIGNAL_MAP = {
    "fan_in":              ("🕸️", "Graph Topology", "#FFB800", "Multiple sources funnelling funds into this account"),
    "fan_out":             ("🕸️", "Graph Topology", "#FFB800", "Funds dispersed to many different recipients"),
    "cycle":               ("🕸️", "Graph Topology", "#FF2B2B", "Circular transaction loop detected in network"),
    "high_centrality":     ("🕸️", "Graph Topology", "#FFB800", "Account is a high-traffic hub in the transaction graph"),
    "bridge":              ("🕸️", "Graph Topology", "#FFB800", "Account bridges otherwise disconnected communities"),
    "cyber":               ("🔐", "Behavioral",      "#FFB800", "Suspicious device or IP behaviour detected"),
    "login":               ("🔐", "Behavioral",      "#FFB800", "Anomalous login pattern or credential activity"),
    "travel":              ("🔐", "Behavioral",      "#FF2B2B", "Impossible travel — access from geographically distant locations"),
    "risk_esc":            ("⏰", "Temporal",         "#FF2B2B", "Risk score escalating over time"),
    "risk_escalation":     ("⏰", "Temporal",         "#FF2B2B", "Risk score escalating over time"),
    "temporal_concentration": ("⏰", "Temporal",      "#FFB800", "Transactions clustered in short time bursts"),
    "volume":              ("⏰", "Temporal",         "#FFB800", "Unusual transaction volume spike"),
    "structuring":         ("⏰", "Temporal",         "#FF2B2B", "Transactions structured to avoid reporting thresholds"),
    "lstm":                ("🔗", "LSTM",             "#00FFFF", "Sequence model flagged abnormal transaction pattern"),
    "emerging":            ("🔗", "LSTM",             "#00FFFF", "New high-probability link predicted to emerge"),
    "emerging_link":       ("🔗", "LSTM",             "#00FFFF", "New high-probability link predicted to emerge"),
    "link_pred":           ("🔗", "LSTM",             "#00FFFF", "LSTM predicted likely future connection"),
    "gbdt":                ("📈", "GBDT",             "#FFB800", "Gradient boosting model scored transaction as high risk"),
    "gnn":                 ("🧠", "GNN",              "#00FFFF", "Graph neural network classified node as suspicious"),
}


def _signal_meta(sig: str):
    sig_low = sig.lower()
    # Exact match first
    if sig_low in _PHASE_SIGNAL_MAP:
        return _PHASE_SIGNAL_MAP[sig_low]
    # Partial keyword match
    for key, meta in _PHASE_SIGNAL_MAP.items():
        if key in sig_low:
            return meta
    return ("⚠️", "Unknown", "#aaaaaa", "Unclassified detection signal")


def _parse_signals(signals_raw: str) -> list[dict]:
    """Deduplicate signals, count occurrences, and group by phase."""
    from collections import Counter
    if not signals_raw or signals_raw in ("", "nan"):
        return []
    parts = [s.strip() for s in signals_raw.split("|") if s.strip() and s.strip() != "nan"]
    counts = Counter(parts)
    total = len(parts)
    result = []
    for sig, count in counts.most_common():
        icon, phase, color, desc = _signal_meta(sig)
        result.append({
            "signal": sig,
            "count": count,
            "pct": count / total * 100,
            "icon": icon,
            "phase": phase,
            "color": color,
            "desc": desc,
        })
    return result


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_sim() -> pd.DataFrame | None:
    for path in ["simulation_pipeline_results.csv", "consolidated_risk_scores.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@st.cache_data(ttl=60)
def _load_tx() -> pd.DataFrame | None:
    if not os.path.exists("transactions.csv"):
        return None
    df = pd.read_csv("transactions.csv")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Account Controls")
    date_from = st.date_input("Timeline From",
                               datetime.utcnow().date() - timedelta(days=365))
    date_to   = st.date_input("Timeline To", datetime.utcnow().date())

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Account Detail")
st.caption(
    "Deep-dive into a single account: risk breakdown, active signals, "
    "transaction timeline, counterparty network, and alert management."
)

# ── Load data ──────────────────────────────────────────────────────────────────
sim_df = _load_sim()
tx_df  = _load_tx()

if sim_df is None:
    st.warning(
        "No simulation results found. "
        "Run `python pipeline_simulation.py` first."
    )
    st.stop()

acc_col   = "account_id" if "account_id" in sim_df.columns else sim_df.columns[0]
accounts  = sorted(sim_df[acc_col].dropna().astype(str).unique().tolist())
all_states = get_all_states()


def _acc_label(acc: str) -> str:
    st_info  = all_states.get(acc, {})
    icon     = STATE_ICONS.get(st_info.get("state", "Unreviewed"), "")
    tier_tag = ""
    if "risk_level" in sim_df.columns:
        row = sim_df[sim_df[acc_col].astype(str) == acc]
        if len(row):
            t = str(row.iloc[0]["risk_level"])
            sym = TIER_SYMBOLS.get(t, "")
            tier_tag = f"[{sym} {t}] " if sym else f"[{t}] "
    return f"{icon}{tier_tag}{acc}"


# ── Account selector ───────────────────────────────────────────────────────────
selected = st.selectbox(
    "Select Account",
    options=["— select an account —"] + accounts,
    format_func=lambda a: a if a == "— select an account —" else _acc_label(a),
)

# ── Overview table when no account selected ────────────────────────────────────
if selected == "— select an account —":
    st.markdown("### 📋 All Accounts Overview")
    st.caption("Select an account above for the full investigation view.")

    rows = []
    for _, row in sim_df.iterrows():
        acc = str(row.get(acc_col, ""))
        st_info = all_states.get(acc, {})
        rows.append({
            "Account":       acc,
            "Risk Level":    row.get("risk_level", "—"),
            "Score":         round(float(row.get("score", 0)), 3)
                             if "score" in row and pd.notna(row.get("score")) else "—",
            "Alert Status":  st_info.get("state", "Unreviewed"),
            "Last Updated":  st_info.get("updated_at", "—"),
        })
    ov_df = pd.DataFrame(rows)

    def _style_ov(val):
        c = TIER_COLORS.get(str(val), "")
        return f"color:{c};font-weight:600" if c else ""

    styled_ov = (
        ov_df.style.map(_style_ov, subset=["Risk Level"])
        if "Risk Level" in ov_df.columns
        else ov_df
    )
    st.dataframe(styled_ov, use_container_width=True, hide_index=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Single account view
# ═══════════════════════════════════════════════════════════════════════════════
acc_rows = sim_df[sim_df[acc_col].astype(str) == selected]
if acc_rows.empty:
    st.error(f"Account `{selected}` not found in results.")
    st.stop()

acc_data    = acc_rows.iloc[0]
risk_level  = str(acc_data.get("risk_level", "CLEAN"))
risk_score  = float(acc_data.get("score", 0)) if "score" in acc_data and pd.notna(acc_data.get("score")) else 0.0
signals_raw = str(acc_data.get("signals", "")) if "signals" in acc_data else ""
tier_color  = TIER_COLORS.get(risk_level, "#888")
tier_sym    = TIER_SYMBOLS.get(risk_level, "●")

# ── Top row: score | signals | alert management ────────────────────────────────
col_score, col_sigs, col_alert = st.columns([1, 2, 1])

with col_score:
    bar_pct = int(risk_score * 100)
    st.markdown(
        f'<div style="background:rgba(20,20,25,0.7);border:2px solid {tier_color}bb;'
        f'border-radius:16px;padding:24px;text-align:center">'
        f'<div style="font-size:0.78rem;color:#aaa;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin-bottom:6px">Risk Level</div>'
        f'<div style="font-size:2.6rem;font-weight:700;color:{tier_color};line-height:1">'
        f'{tier_sym} {risk_level}</div>'
        f'<div style="color:#bbb;font-size:1rem;margin-top:6px">'
        f'Score: <b style="color:#fff">{risk_score:.3f}</b></div>'
        f'<div style="margin-top:14px;background:#2a2a2a;border-radius:8px;height:7px">'
        f'<div style="background:{tier_color};border-radius:8px;height:7px;'
        f'width:{bar_pct}%;transition:width 0.4s ease"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with col_sigs:
    st.markdown("**🔬 Active Signals**")
    parsed_signals = _parse_signals(signals_raw)
    if parsed_signals:
        total_signal_hits = sum(s["count"] for s in parsed_signals)
        st.caption(f"{len(parsed_signals)} distinct signal type{'s' if len(parsed_signals) != 1 else ''} · {total_signal_hits} total hits")

        # Group by phase
        from collections import defaultdict
        by_phase = defaultdict(list)
        for s in parsed_signals:
            by_phase[s["phase"]].append(s)

        for phase, sigs in by_phase.items():
            phase_icon = sigs[0]["icon"]
            phase_color = sigs[0]["color"]
            phase_hits = sum(s["count"] for s in sigs)
            # Phase header
            st.markdown(
                f'<div style="margin-top:8px;margin-bottom:2px;font-size:0.72rem;'
                f'text-transform:uppercase;letter-spacing:0.1em;color:{phase_color};'
                f'font-weight:600">{phase_icon} {phase} &nbsp;'
                f'<span style="color:#888">({phase_hits} hits)</span></div>',
                unsafe_allow_html=True,
            )
            for s in sigs:
                bar_w = min(100, int(s["pct"] * 2))  # visual bar scaled to 50% max
                count_badge = (
                    f'<span style="background:{s["color"]}22;color:{s["color"]};'
                    f'border-radius:4px;padding:1px 6px;font-size:0.72rem;'
                    f'font-weight:700;margin-left:6px">×{s["count"]}</span>'
                    if s["count"] > 1 else ""
                )
                st.markdown(
                    f'<div style="padding:5px 10px;margin:2px 0;border-radius:6px;'
                    f'background:rgba(0,0,0,0.35);border-left:3px solid {s["color"]};">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:#ccc;font-size:0.8rem">{s["signal"]}{count_badge}</span>'
                    f'</div>'
                    f'<div style="color:#999;font-size:0.71rem;margin-top:2px">{s["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.success("No alert signals — this account is CLEAN.")

with col_alert:
    st.markdown("**📋 Alert Management**")
    alert_info  = get_alert_state(selected)
    current_st  = alert_info.get("state", "Unreviewed")
    new_state   = st.selectbox(
        "Status",
        STATES,
        index=STATES.index(current_st) if current_st in STATES else 0,
        key="acct_state",
    )
    note_val = st.text_area(
        "Investigation Notes",
        value=alert_info.get("note", ""),
        height=110,
        key="acct_note",
        placeholder="Add investigation notes, reference IDs, or disposition rationale…",
    )
    if st.button("💾 Save", type="primary", key="acct_save"):
        set_alert_state(selected, new_state, note_val)
        st.success(f"Saved: {new_state}", icon="✅")
        # Update the local cache so the selector label refreshes
        all_states[selected] = {"state": new_state, "note": note_val,
                                  "updated_at": datetime.utcnow().isoformat()}
    if alert_info.get("updated_at"):
        st.caption(f"Last updated: {alert_info['updated_at'][:19]} UTC")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Transaction Timeline ────────────────────────────────────────────────────────
st.markdown("### 📅 Transaction Timeline")

if tx_df is not None and "source" in tx_df.columns and "target" in tx_df.columns:
    acc_tx = tx_df[
        (tx_df["source"].astype(str) == selected) |
        (tx_df["target"].astype(str) == selected)
    ].copy()

    if "timestamp" in acc_tx.columns:
        acc_tx = acc_tx.dropna(subset=["timestamp"])
        acc_tx = acc_tx[
            (acc_tx["timestamp"].dt.date >= date_from) &
            (acc_tx["timestamp"].dt.date <= date_to)
        ]

    if len(acc_tx) > 0:
        acc_tx["direction"] = acc_tx.apply(
            lambda r: "Outgoing" if str(r["source"]) == selected else "Incoming",
            axis=1,
        )

        fig_tl = go.Figure()
        for direction, color in [("Incoming", "#00FF94"), ("Outgoing", "#FF2B2B")]:
            d = acc_tx[acc_tx["direction"] == direction]
            if len(d) > 0 and "amount" in d.columns:
                fig_tl.add_trace(go.Bar(
                    x=d["timestamp"] if "timestamp" in d.columns else d.index,
                    y=d["amount"],
                    name=direction,
                    marker_color=color,
                    opacity=0.8,
                    hovertemplate=f"{direction}: $%{{y:,.0f}}<br>%{{x}}<extra></extra>",
                ))

        fig_tl.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, title="Date"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.12)",
                       title="Amount ($)"),
            legend=dict(orientation="h", y=1.12),
            font=dict(family="system-ui"),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        n_in  = len(acc_tx[acc_tx["direction"] == "Incoming"])
        n_out = len(acc_tx[acc_tx["direction"] == "Outgoing"])
        total_amt = acc_tx.get("amount", pd.Series(dtype=float)).sum() if "amount" in acc_tx.columns else 0
        st.caption(
            f"{len(acc_tx)} transactions in selected period  |  "
            f"Incoming: {n_in}  |  Outgoing: {n_out}  |  "
            f"Total volume: ${total_amt:,.0f}"
        )
    else:
        st.info("No transactions found for this account in the selected date range.")
else:
    st.info(
        "Transaction data not available. "
        "Run `python main.py` to generate `transactions.csv`."
    )

# ── Counterparty Network ───────────────────────────────────────────────────────
st.markdown("### 🕸️ Counterparty Network")

if (tx_df is not None
        and "source" in tx_df.columns
        and "target" in tx_df.columns):

    acc_tx_all = tx_df[
        (tx_df["source"].astype(str) == selected) |
        (tx_df["target"].astype(str) == selected)
    ]

    if len(acc_tx_all) > 0:
        G_local = nx.DiGraph()
        for _, r in acc_tx_all.iterrows():
            G_local.add_edge(
                str(r["source"]), str(r["target"]),
                amount=float(r["amount"]) if "amount" in r and pd.notna(r["amount"]) else 0.0,
            )

        pos = nx.spring_layout(G_local, seed=42)

        # Resolve each node's tier
        node_tiers = {}
        for n in G_local.nodes():
            if str(n) == selected:
                node_tiers[n] = "__SELECTED__"
            else:
                tier = "CLEAN"
                if "risk_level" in sim_df.columns:
                    match = sim_df[sim_df[acc_col].astype(str) == str(n)]
                    if len(match):
                        tier = str(match.iloc[0].get("risk_level", "CLEAN"))
                node_tiers[n] = tier

        # Edges — visible on dark bg
        ex, ey = [], []
        for u, v in G_local.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex += [x0, x1, None]; ey += [y0, y1, None]

        edge_tr = go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.15)"),
            hoverinfo="none", showlegend=False,
        )

        # One trace per tier for a proper legend
        TIER_DISPLAY = {
            "__SELECTED__": ("⭐ Selected",  "#00FFFF", 20, "#ffffff"),
            "CRITICAL":     ("▲ Critical",   "#FF2B2B", 14, "#FF2B2B"),
            "HIGH":         ("▲ High",        "#FF6B35", 14, "#FF6B35"),
            "MEDIUM":       ("◆ Medium",      "#FFB800", 12, "#FFB800"),
            "LOW":          ("✓ Low",         "#00FF94", 10, "#00FF94"),
            "CLEAN":        ("○ Clean",       "#9090a0", 10, "#bbbbcc"),
        }

        traces = [edge_tr]
        for tier_key, (label, color, size, text_color) in TIER_DISPLAY.items():
            nodes_in_tier = [n for n, t in node_tiers.items() if t == tier_key]
            if not nodes_in_tier:
                continue
            border_color = "#ffffff" if tier_key == "__SELECTED__" else "#3a3a4a"
            border_width = 2 if tier_key == "__SELECTED__" else 1
            hover = [
                f"⭐ {n} (selected)" if tier_key == "__SELECTED__"
                else f"{TIER_SYMBOLS.get(tier_key, '●')} {n} [{tier_key}]"
                for n in nodes_in_tier
            ]
            traces.append(go.Scatter(
                x=[pos[n][0] for n in nodes_in_tier],
                y=[pos[n][1] for n in nodes_in_tier],
                mode="markers+text",
                name=label,
                text=[str(n) for n in nodes_in_tier],
                textposition="top center",
                textfont=dict(size=9, color=text_color),
                marker=dict(
                    size=size, color=color,
                    line=dict(width=border_width, color=border_color),
                ),
                hovertext=hover, hoverinfo="text",
            ))

        fig_local = go.Figure(data=traces)
        fig_local.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,5,10,1)",
            height=420, margin=dict(l=0, r=0, t=10, b=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                orientation="h", y=-0.08, x=0.5, xanchor="center",
                font=dict(size=11, color="#ccc"),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(family="system-ui"),
        )
        st.plotly_chart(fig_local, use_container_width=True)
    else:
        st.info("No transaction connections found for this account.")
else:
    st.info("Transaction data not available for the counterparty graph.")
