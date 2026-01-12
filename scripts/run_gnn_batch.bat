@echo off
echo Starting GNN Embedding Generation Batch Job...
cd C:\Users\SBaskar\aml-topology\aml-topology
python scripts\generate_gnn_embeddings.py
echo.
echo Batch job completed.
pause
