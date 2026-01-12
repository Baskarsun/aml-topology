import networkx as nx
import matplotlib.pyplot as plt
import time
from typing import Iterable, Callable, Optional


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
