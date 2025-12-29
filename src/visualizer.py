import networkx as nx
import matplotlib.pyplot as plt

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
