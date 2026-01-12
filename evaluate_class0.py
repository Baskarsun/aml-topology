"""
Evaluate Class 0 (Legitimate) Accuracy of the GNN
==================================================
Checks if the retrained GNN is "too paranoid" by measuring
false positive rate on legitimate nodes.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from src.gnn_trainer import (
    GraphSage, 
    build_node_features, 
    generate_synthetic_graph, 
    adjacency_sparse_from_nx, 
    graph_to_tx_df
)


def evaluate_class0_accuracy(gnn_path: str = 'models/gnn_adversarial.pt', seed: int = 123):
    """Evaluate Class 0 accuracy on a clean baseline graph."""
    
    print("=" * 50)
    print("CLASS 0 (LEGITIMATE) ACCURACY EVALUATION")
    print("=" * 50)
    
    # Load retrained GNN
    print(f"\nLoading GNN from {gnn_path}...")
    gnn = GraphSage(in_feats=12, hidden=64, out_feats=32)
    gnn.load_state_dict(torch.load(gnn_path, map_location='cpu'))
    gnn.eval()
    
    # Generate clean baseline graph
    print("Generating clean baseline graph (500 nodes)...")
    G, labels, profiles = generate_synthetic_graph(
        num_nodes=500, 
        mule_fraction=0.02,  # Only 2% mules
        seed=seed
    )
    
    # Build features
    print("Building node features...")
    df_tx = graph_to_tx_df(G)
    features, node_index, _ = build_node_features(G, df_tx)
    adj = adjacency_sparse_from_nx(G, node_index)
    
    # Get predictions
    print("Running GNN predictions...")
    with torch.no_grad():
        probs = gnn(torch.FloatTensor(features), adj).numpy()
    
    # Build ground truth labels
    idx_to_node = {i: n for n, i in node_index.items()}
    gt_labels = np.array([labels.get(idx_to_node[i], 0) for i in range(len(node_index))])
    
    # Calculate per-class metrics
    class0_mask = gt_labels == 0  # Legitimate
    class1_mask = gt_labels == 1  # Suspicious
    
    preds = (probs > 0.5).astype(int)
    
    # Class 0 metrics
    class0_correct = ((preds == 0) & class0_mask).sum()
    class0_total = class0_mask.sum()
    class0_acc = class0_correct / class0_total if class0_total > 0 else 0
    
    # Class 1 metrics
    class1_correct = ((preds == 1) & class1_mask).sum()
    class1_total = class1_mask.sum()
    class1_acc = class1_correct / class1_total if class1_total > 0 else 0
    
    # False positive rate (legitimate flagged as suspicious)
    false_positives = ((preds == 1) & class0_mask).sum()
    fpr = false_positives / class0_total if class0_total > 0 else 0
    
    # Print results
    print("\n" + "=" * 50)
    print("CLASS 0 (LEGITIMATE NODES)")
    print("=" * 50)
    print(f"Total Class 0 nodes:     {class0_total}")
    print(f"Correctly classified:    {class0_correct}")
    print(f"Class 0 Accuracy:        {class0_acc:.1%}")
    print(f"False Positive Rate:     {fpr:.1%}")
    
    print("\n" + "=" * 50)
    print("CLASS 1 (SUSPICIOUS NODES)")
    print("=" * 50)
    print(f"Total Class 1 nodes:     {class1_total}")
    print(f"Correctly classified:    {class1_correct}")
    print(f"Class 1 Accuracy:        {class1_acc:.1%}")
    
    # Overall
    overall_acc = (class0_correct + class1_correct) / (class0_total + class1_total)
    print(f"\nOverall Accuracy:        {overall_acc:.1%}")
    
    # Decision
    print("\n" + "=" * 50)
    print("DECISION")
    print("=" * 50)
    
    if class0_acc >= 0.95:
        print("✅ Class 0 Accuracy >= 95%: PROCEED")
        print("   GNN is well-calibrated")
        decision = "PROCEED"
    elif class0_acc >= 0.90:
        print("⚠️ Class 0 Accuracy 90-95%: BORDERLINE")
        print("   Consider slightly increasing organic weight")
        decision = "BORDERLINE"
    else:
        print("❌ Class 0 Accuracy < 90%: GNN IS TOO PARANOID!")
        print("   Action Required: Increase organic data weight in training")
        print("   Recommended: 60% Organic / 20% Old Fraud / 20% New Fraud")
        decision = "TOO_PARANOID"
    
    return {
        'class0_accuracy': class0_acc,
        'class1_accuracy': class1_acc,
        'false_positive_rate': fpr,
        'overall_accuracy': overall_acc,
        'decision': decision,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn', default='models/gnn_adversarial.pt')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    evaluate_class0_accuracy(args.gnn, args.seed)
