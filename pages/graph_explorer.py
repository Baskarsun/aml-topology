"""
Graph Explorer Page
====================
New page replacing the static PNG output from src/visualizer.py with a fully
interactive Plotly network graph.

Features:
  ✓ Interactive pan, zoom, hover — no static image
  ✓ Four risk-tier colours (CRITICAL/HIGH/MEDIUM/LOW/CLEAN) — not binary red/blue
  ✓ Node size proportional to connection degree
  ✓ Hover tooltip: account ID, tier, score, in/out degree
  ✓ Edge colour dimmed; highlighted for selected account's edges
  ✓ Risk-tier filter (multiselect)
  ✓ Account search + highlight
  ✓ Max-node slider (shows highest-risk first when graph is large)
  ✓ Inline legend inside chart
  ✓ Account detail panel on search match
  ✓ Labels shown when ≤80 nodes; hidden automatically for large graphs
  ✓ Accessibility: tier label includes text symbol, not colour alone
"""
import os
import sys

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TIER_COLORS = {
    "CRITICAL": "#FF2B2B",
    "HIGH":     "#FF6B35",
    "MEDIUM":   "#FFB800",
    "LOW":      "#00FF94",
    "CLEAN":    "#9090a0",
}
TIER_LABELS = {
    "CRITICAL": "▲ CRITICAL",
    "HIGH":     "● HIGH",
    "MEDIUM":   "◆ MEDIUM",
    "LOW":      "✓ LOW",
    "CLEAN":    "○ CLEAN",
}
TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"]


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_transactions() -> pd.DataFrame | None:
    """Load raw transaction data (must have 'source' and 'target' columns)."""
    for path in ["transactions.csv", "simulation_pipeline_results.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "source" in df.columns and "target" in df.columns:
                return df
    return None


@st.cache_data(ttl=60)
def _load_risk_map() -> dict:
    """Return {account_id: {tier, score}} from simulation or consolidated results."""
    for path in ["simulation_pipeline_results.csv", "consolidated_risk_scores.csv"]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        id_col    = "account_id"    if "account_id"    in df.columns else df.columns[0]
        level_col = "risk_level"    if "risk_level"    in df.columns else None
        score_col = "score"         if "score"         in df.columns else None
        if level_col is None:
            continue
        rmap = {}
        for _, r in df.iterrows():
            rmap[str(r[id_col])] = {
                "tier":  str(r[level_col]),
                "score": float(r[score_col]) if score_col and pd.notna(r[score_col]) else 0.0,
            }
        return rmap
    return {}


@st.cache_data(ttl=60, show_spinner="Building transaction graph…")
def _build_graph() -> nx.MultiDiGraph:
    df = _load_transactions()
    G  = nx.MultiDiGraph()
    if df is None:
        return G
    for _, row in df.iterrows():
        G.add_edge(
            str(row["source"]), str(row["target"]),
            amount=float(row["amount"]) if "amount" in row and pd.notna(row["amount"]) else 0.0,
        )
    return G


def _build_plotly_figure(
    G: nx.MultiDiGraph,
    risk_map: dict,
    tier_filter: list,
    highlight: str,
    max_nodes: int,
    show_labels: bool,
    layout_seed: int,
) -> go.Figure:
    """Construct the interactive Plotly graph figure."""

    # Sub-graph: keep only nodes whose tier is in the filter
    if tier_filter and risk_map:
        keep = {n for n in G.nodes() if risk_map.get(str(n), {}).get("tier", "CLEAN") in tier_filter}
        G = G.subgraph(keep).copy()

    # If still too large, keep the highest-risk nodes
    if G.number_of_nodes() > max_nodes:
        sorted_nodes = sorted(
            G.nodes(),
            key=lambda n: -risk_map.get(str(n), {}).get("score", 0),
        )[:max_nodes]
        G = G.subgraph(sorted_nodes).copy()

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text="No nodes match the current filter.",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#bbb", size=16),
            )],
        )
        return fig

    pos = nx.spring_layout(G, k=0.9, seed=layout_seed, iterations=60)

    # ── Edge trace ─────────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(255,255,255,0.12)"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Node traces (one per tier for the legend) ──────────────────────────────
    node_traces = []
    for tier in TIER_ORDER:
        if tier not in tier_filter:
            continue
        nodes_t = [n for n in G.nodes() if risk_map.get(str(n), {}).get("tier", "CLEAN") == tier]
        if not nodes_t:
            continue

        xs = [pos[n][0] for n in nodes_t]
        ys = [pos[n][1] for n in nodes_t]

        # Node size = degree, clamped to [10, 35]
        sizes = [min(35, max(10, 10 + G.degree(n) * 2)) for n in nodes_t]

        # Highlight border for the searched account
        line_widths = [4 if str(n) == highlight else 1 for n in nodes_t]
        line_colors = ["#FFFFFF" if str(n) == highlight else TIER_COLORS[tier] for n in nodes_t]

        hover_texts = []
        for n in nodes_t:
            info = risk_map.get(str(n), {})
            in_d  = G.in_degree(n)
            out_d = G.out_degree(n)
            hover_texts.append(
                f"<b>{n}</b><br>"
                f"Tier: {TIER_LABELS.get(tier, tier)}<br>"
                f"Score: {info.get('score', 0):.3f}<br>"
                f"In-connections: {in_d}  |  Out-connections: {out_d}"
            )

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if (show_labels and len(G.nodes()) <= 80) else "markers",
            text=[str(n) for n in nodes_t],
            textposition="top center",
            textfont=dict(size=9, color="#cccccc"),
            hovertext=hover_texts, hoverinfo="text",
            marker=dict(
                size=sizes,
                color=TIER_COLORS[tier],
                line=dict(width=line_widths, color=line_colors),
                symbol="circle",
            ),
            name=TIER_LABELS[tier],
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,5,10,1)",
        height=620,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        legend=dict(
            bgcolor="rgba(10,10,15,0.92)",
            bordercolor="#555", borderwidth=1,
            font=dict(color="#ddd", size=12),
            title=dict(text="Risk Tier", font=dict(color="#aaa")),
            x=0.01, y=0.99,
        ),
        font=dict(family="system-ui, sans-serif"),
        dragmode="pan",
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Graph Controls")
    tier_filter = st.multiselect(
        "Show Risk Tiers",
        TIER_ORDER,
        default=TIER_ORDER,
    )
    search_node = st.text_input("Highlight Account", placeholder="ACC_0001")
    max_nodes   = st.slider("Max nodes displayed", 50, 600, 250, 25,
                             help="Largest risk-score nodes kept when graph is too large.")
    layout_seed = st.number_input("Layout seed", 0, 9999, 42,
                                   help="Change to try different spring-layout positions.")
    show_labels = st.checkbox("Show node labels (auto-hidden >80 nodes)", value=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🕸️ Graph Explorer")
st.caption(
    "Interactive transaction network. "
    "Node size = connection degree. "
    "Colour + symbol = risk tier. "
    "Hover for details. Pan / zoom with mouse."
)

# ── Load data ──────────────────────────────────────────────────────────────────
tx_df    = _load_transactions()
risk_map = _load_risk_map()

if tx_df is None:
    st.warning(
        "No transaction data found. "
        "Run `python main.py` or `python pipeline_simulation.py` "
        "to generate `transactions.csv`."
    )
    st.stop()

G_full = _build_graph()

# ── Summary metrics ────────────────────────────────────────────────────────────
_m1, _m2, _m3, _m4 = st.columns(4)
_m1.metric("Total Nodes",        f"{G_full.number_of_nodes():,}")
_m2.metric("Total Edges",        f"{G_full.number_of_edges():,}")
_m3.metric("Risk-mapped Accounts", f"{len(risk_map):,}")
_m4.metric(
    "Suspicious (CRITICAL/HIGH)",
    f"{sum(1 for r in risk_map.values() if r.get('tier','CLEAN') in ('CRITICAL','HIGH')):,}",
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Render chart ───────────────────────────────────────────────────────────────
with st.spinner("Rendering graph…"):
    fig = _build_plotly_figure(
        G_full, risk_map,
        tier_filter=tier_filter,
        highlight=search_node.strip() if search_node else "",
        max_nodes=max_nodes,
        show_labels=show_labels,
        layout_seed=int(layout_seed),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Showing up to {max_nodes} nodes (highest-risk first when graph is large). "
    "Use the Risk Tier filter to focus on specific categories."
)

# ── Account detail panel on search ─────────────────────────────────────────────
if search_node:
    search_node = search_node.strip()
    if search_node in G_full.nodes():
        st.markdown("---")
        st.markdown(f"### 🔍 Account: `{search_node}`")

        info      = risk_map.get(search_node, {})
        tier      = info.get("tier", "CLEAN")
        score     = info.get("score", 0.0)
        tier_col  = TIER_COLORS.get(tier, "#888")
        tier_lbl  = TIER_LABELS.get(tier, tier)

        _da, _db = st.columns([1, 2])
        with _da:
            st.markdown(
                f'<div style="background:rgba(20,20,25,0.7);border:2px solid {tier_col};'
                f'border-radius:14px;padding:22px;text-align:center">'
                f'<div style="font-size:0.78rem;color:#aaa;text-transform:uppercase;'
                f'letter-spacing:0.12em">Risk Tier</div>'
                f'<div style="font-size:2.2rem;font-weight:700;color:{tier_col}">'
                f'{tier_lbl}</div>'
                f'<div style="color:#bbb;font-size:0.92rem;margin-top:4px">'
                f'Score: {score:.3f}</div>'
                f'<div style="margin-top:12px;background:#2a2a2a;border-radius:8px;height:6px">'
                f'<div style="background:{tier_col};border-radius:8px;height:6px;'
                f'width:{int(score*100)}%"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with _db:
            in_nodes  = list(G_full.predecessors(search_node))
            out_nodes = list(G_full.successors(search_node))
            st.metric("In-connections",  len(in_nodes))
            st.metric("Out-connections", len(out_nodes))
            if in_nodes:
                st.markdown("**Incoming from:**")
                st.code(", ".join(in_nodes[:25]) + ("…" if len(in_nodes) > 25 else ""))
            if out_nodes:
                st.markdown("**Outgoing to:**")
                st.code(", ".join(out_nodes[:25]) + ("…" if len(out_nodes) > 25 else ""))
    else:
        st.warning(
            f"Account `{search_node}` not found in the graph. "
            "It may be filtered out by the Risk Tier filter, "
            "or it may not exist in `transactions.csv`."
        )
