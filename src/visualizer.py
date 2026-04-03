import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from typing import Iterable, Callable, Optional, Dict


def plot_graph(G, suspicious_nodes=None, filename="transaction_network.png"):
    plt.figure(figsize=(12, 12))

    # Create a layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)

    # Default node colors
    node_colors = ['lightblue'] * len(G.nodes())
    node_sizes = [300] * len(G.nodes())

    # Map nodes to indices to update colors
    node_list = list(G.nodes())

    if suspicious_nodes:
        for i, node in enumerate(node_list):
            if node in suspicious_nodes:
                node_colors[i] = 'red'  # Highlight suspicious nodes
                node_sizes[i] = 600

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=True)

    # Draw labels (optional, can be cluttered)
    # nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("AML Transaction Network Visualization")
    plt.axis('off')

    plt.savefig(filename)
    print(f"Graph visualization saved to {filename}")
    plt.close()


def live_plot(graph_generator: Iterable[nx.Graph],
              suspicious_nodes_getter: Optional[Callable[[nx.Graph], Iterable]] = None,
              node_info_getter: Optional[Callable[[nx.Graph, str], dict]] = None,
              update_interval: float = 0.5,
              figsize=(12, 12),
              layout_func: Optional[Callable[[nx.Graph], dict]] = None,
              max_frames: Optional[int] = None):
    """
    Live-plot a stream/sequence of NetworkX graphs using matplotlib.

    Parameters:
    - graph_generator: an iterable yielding NetworkX Graph/MultiDiGraph objects.
    - suspicious_nodes_getter: optional callable that accepts a graph and
      returns an iterable of node identifiers to highlight (e.g. ['ACC_0001']).
    - update_interval: seconds to wait between frames (uses `plt.pause`).
    - layout_func: optional callable to compute layout dict; if None, the
      layout is computed from the first graph using `spring_layout` and reused.
    - max_frames: optional limit of frames to display.

    Notes:
    - This is suitable for small-to-moderate graphs and quick debugging.
    - For large graphs or production dashboards, consider Dash/Streamlit/pyvis.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=figsize)

    pos = None
    frame = 0

    try:
        def default_node_info(G, node):
            # compute simple stats for a node in a MultiDiGraph
            try:
                in_edges = G.in_edges(node, data=True)
                out_edges = G.out_edges(node, data=True)
            except Exception:
                in_edges = []
                out_edges = []

            total_in = sum([d.get('amount', 0) for _, _, d in in_edges])
            total_out = sum([d.get('amount', 0) for _, _, d in out_edges])
            return {
                'node': str(node),
                'in_degree': G.in_degree(node) if hasattr(G, 'in_degree') else 0,
                'out_degree': G.out_degree(node) if hasattr(G, 'out_degree') else 0,
                'total_in': round(total_in, 2),
                'total_out': round(total_out, 2),
                'balance_change': round(total_in - total_out, 2)
            }

        node_info_getter = node_info_getter or default_node_info

        # helper to update annotation text/position
        def update_annot(ind, node_list, pos):
            idx = ind
            node = node_list[idx]
            info = node_info_getter(current_graph, node)
            text_lines = [f"{k}: {v}" for k, v in info.items()]
            annot.set_text('\n'.join(text_lines))
            annot.xy = pos[node]
            annot.get_bbox_patch().set_facecolor('white')
            annot.get_bbox_patch().set_alpha(0.9)

        # motion event handler
        def on_move(event):
            if event.inaxes != ax:
                try:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                except Exception:
                    pass
                return

            # transform node positions to display coords
            display_pos = {n: ax.transData.transform(p) for n, p in pos.items()}
            x, y = event.x, event.y
            # find nearest node
            nearest = None
            nearest_dist = float('inf')
            for i, n in enumerate(node_list):
                px, py = display_pos.get(n, (None, None))
                if px is None:
                    continue
                dist = (px - x) ** 2 + (py - y) ** 2
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (i, n)

            # threshold in pixels (squared)
            threshold = 100  # ~10px radius
            if nearest and nearest_dist <= threshold:
                idx, _ = nearest
                update_annot(idx, node_list, pos)
                try:
                    annot.set_visible(True)
                except Exception:
                    pass
                fig.canvas.draw_idle()
            else:
                try:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
                except Exception:
                    pass

        # will be set per-frame
        current_graph = None
        node_list = []
        for G in graph_generator:
            if G is None:
                time.sleep(update_interval)
                continue

            # compute or reuse layout. If new nodes appear, run a short
            # spring_layout pass with the previous positions as initialization
            # so all nodes get positions and the layout evolves smoothly.
            if pos is None:
                if layout_func:
                    pos = layout_func(G)
                else:
                    pos = nx.spring_layout(G, k=0.15, iterations=20)
            else:
                # ensure all nodes have positions; spring_layout will add
                # missing nodes when given `pos` as initial state
                if layout_func:
                    # if user provided a layout_func, prefer calling it each frame
                    pos = layout_func(G)
                else:
                    # run fewer iterations to cheaply adjust layout
                    pos = nx.spring_layout(G, k=0.15, iterations=5, pos=pos)

            ax.clear()

            # create annotation after clearing axes so it is attached to current axes
            annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            # prepare node visuals
            node_list = list(G.nodes())
            node_colors = ['lightblue'] * len(node_list)
            node_sizes = [300] * len(node_list)

            suspicious_nodes = None
            if suspicious_nodes_getter:
                try:
                    suspicious_nodes = set(suspicious_nodes_getter(G) or [])
                except Exception:
                    suspicious_nodes = set()

            if suspicious_nodes:
                for i, n in enumerate(node_list):
                    if n in suspicious_nodes:
                        node_colors[i] = 'red'
                        node_sizes[i] = 600

            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
            nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.6, arrows=True, ax=ax)

            ax.set_title("Live AML Transaction Network")
            ax.axis('off')

            fig.canvas.draw()
            plt.pause(update_interval)

            # update handler state for hover
            current_graph = G

            # connect the motion handler once
            if not hasattr(on_move, "connected"):
                cid = fig.canvas.mpl_connect('motion_notify_event', on_move)
                on_move.connected = cid

            frame += 1
            if max_frames and frame >= max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Plotly graph (replaces static PNG for dashboard / Streamlit use)
# ──────────────────────────────────────────────────────────────────────────────

#: Risk-tier colour map — 5 tiers, each with a distinct colour AND a text symbol
#: so the visualisation is accessible to colour-blind users.
TIER_COLORS = {
    "CRITICAL": "#FF2B2B",
    "HIGH":     "#FF6B35",
    "MEDIUM":   "#FFB800",
    "LOW":      "#00FF94",
    "CLEAN":    "#555555",
}
TIER_SYMBOLS = {
    "CRITICAL": "▲",
    "HIGH":     "●",
    "MEDIUM":   "◆",
    "LOW":      "✓",
    "CLEAN":    "○",
}
_TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"]


def build_plotly_graph(
    G: nx.Graph,
    node_risks: Optional[Dict[str, dict]] = None,
    suspicious_nodes=None,
    highlight_node: Optional[str] = None,
    show_labels: bool = True,
    height: int = 600,
    layout_seed: int = 42,
):
    """
    Build an **interactive** Plotly figure from a NetworkX graph.

    This replaces the static PNG produced by :func:`plot_graph` for contexts
    where interactivity (pan, zoom, hover) is available, such as Streamlit or
    a Jupyter notebook.

    Parameters
    ----------
    G : nx.Graph
        Any NetworkX graph (directed or undirected).
    node_risks : dict, optional
        Mapping ``{node_id: {"tier": str, "score": float}}``.
        Tiers: ``"CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "CLEAN"``.
        When provided, nodes are coloured by tier.
    suspicious_nodes : iterable, optional
        Fallback when *node_risks* is not provided.  Nodes in this set are
        rendered as ``"HIGH"``; all others as ``"CLEAN"``.
    highlight_node : str, optional
        A single node to draw with a bright white border.
    show_labels : bool
        Show node-ID text labels.  Automatically suppressed for graphs with
        more than 80 nodes regardless of this flag.
    height : int
        Chart height in pixels.
    layout_seed : int
        Random seed for the spring-layout algorithm.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for build_plotly_graph. "
            "Install it with: pip install plotly"
        )

    # Build a fallback risk map from suspicious_nodes if node_risks not given
    if node_risks is None:
        suspicious = set(suspicious_nodes or [])
        node_risks = {
            str(n): {"tier": "HIGH" if n in suspicious else "CLEAN", "score": 0.0}
            for n in G.nodes()
        }

    pos = nx.spring_layout(G, k=0.85, seed=layout_seed, iterations=60)

    # ── Edge trace (single trace; no per-edge hover needed here) ───────────────
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="#2a2a2a"),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Node traces — one per tier so Plotly renders a proper legend ───────────
    node_traces = []
    _auto_labels = show_labels and len(G.nodes()) <= 80

    for tier in _TIER_ORDER:
        nodes_t = [n for n in G.nodes()
                   if node_risks.get(str(n), {}).get("tier", "CLEAN") == tier]
        if not nodes_t:
            continue

        xs    = [pos[n][0] for n in nodes_t]
        ys    = [pos[n][1] for n in nodes_t]
        sizes = [min(35, max(10, 10 + G.degree(n) * 2)) for n in nodes_t]

        border_colors = [
            "#FFFFFF" if str(n) == str(highlight_node) else TIER_COLORS[tier]
            for n in nodes_t
        ]
        border_widths = [
            4 if str(n) == str(highlight_node) else 1
            for n in nodes_t
        ]

        hover_texts = []
        for n in nodes_t:
            info  = node_risks.get(str(n), {})
            in_d  = G.in_degree(n)  if G.is_directed() else G.degree(n)
            out_d = G.out_degree(n) if G.is_directed() else G.degree(n)
            hover_texts.append(
                f"<b>{n}</b><br>"
                f"Tier: {TIER_SYMBOLS.get(tier,'')} {tier}<br>"
                f"Score: {info.get('score', 0):.3f}<br>"
                f"In: {in_d}  |  Out: {out_d}"
            )

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if _auto_labels else "markers",
            text=[str(n) for n in nodes_t],
            textposition="top center",
            textfont=dict(size=8, color="#999"),
            hovertext=hover_texts, hoverinfo="text",
            marker=dict(
                size=sizes,
                color=TIER_COLORS[tier],
                line=dict(width=border_widths, color=border_colors),
            ),
            name=f"{TIER_SYMBOLS.get(tier,'')} {tier}",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,5,10,1)",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False),
        legend=dict(
            bgcolor="rgba(10,10,15,0.85)",
            bordercolor="#333", borderwidth=1,
            font=dict(color="#ccc", size=12),
            title=dict(text="Risk Tier", font=dict(color="#666")),
        ),
        font=dict(family="system-ui, sans-serif"),
        dragmode="pan",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Updated static plot_graph — now includes legend and risk-tier colours
# ──────────────────────────────────────────────────────────────────────────────

def plot_graph_with_tiers(
    G: nx.Graph,
    node_risks: Optional[Dict[str, dict]] = None,
    suspicious_nodes=None,
    filename: str = "transaction_network.png",
):
    """
    Enhanced static PNG graph visualisation with:
      - 5-tier risk colouring (not just red vs blue)
      - Node size proportional to degree
      - Edge labels for transaction amounts (when available)
      - Legend identifying each tier by colour AND text symbol
    """
    tier_color_map = {
        "CRITICAL": "red",
        "HIGH":     "orangered",
        "MEDIUM":   "gold",
        "LOW":      "limegreen",
        "CLEAN":    "lightblue",
    }

    if node_risks is None:
        suspicious = set(suspicious_nodes or [])
        node_risks = {
            str(n): {"tier": "HIGH" if n in suspicious else "CLEAN"}
            for n in G.nodes()
        }

    node_list   = list(G.nodes())
    node_colors = [tier_color_map.get(node_risks.get(str(n), {}).get("tier", "CLEAN"), "lightblue")
                   for n in node_list]
    node_sizes  = [max(200, min(800, 200 + G.degree(n) * 40)) for n in node_list]

    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.2, iterations=30, seed=42)

    nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                           node_color=node_colors, node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.4, arrows=True,
                           arrowsize=12, edge_color="#666")
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="#333")

    # Edge amount labels (if the graph carries 'amount' edge data)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if "amount" in data and data["amount"]:
            edge_labels[(u, v)] = f"${data['amount']:,.0f}"
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Legend
    patches = [
        mpatches.Patch(color=col, label=f"{TIER_SYMBOLS.get(tier,'')} {tier}")
        for tier, col in tier_color_map.items()
    ]
    plt.legend(handles=patches, loc="upper left", fontsize=10,
               framealpha=0.8, title="Risk Tier")

    plt.title("AML Transaction Network", fontsize=14, pad=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {filename}")


def graph_generator_from_dataframe(df, step: int = 50, cumulative: bool = True):
    """
    Simple helper: yields incremental NetworkX MultiDiGraph objects built from
    slices of a pandas DataFrame with columns ['source','target','amount','timestamp','type'].

    Parameters:
    - df: pandas DataFrame containing transactions
    - step: number of rows to add per yielded graph
    - cumulative: if True, each graph contains all rows up to the current slice;
      if False, each graph contains only the current slice.
    """
    import math

    n = len(df)
    if n == 0:
        return

    for start in range(0, n, step):
        end = min(start + step, n)
        if cumulative:
            slice_df = df.iloc[:end]
        else:
            slice_df = df.iloc[start:end]

        G = nx.MultiDiGraph()
        for _, row in slice_df.iterrows():
            G.add_edge(row['source'], row['target'], amount=row.get('amount'), timestamp=row.get('timestamp'), type=row.get('type'))

        yield G
