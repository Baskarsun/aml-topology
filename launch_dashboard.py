"""
Quick Launch Script for AML Dashboard Demo

Starts all three components in separate processes:
1. Flask Inference API
2. Transaction Simulator
3. Streamlit Dashboard
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = ['flask', 'streamlit', 'plotly', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def check_models():
    """Check if models are trained."""
    models_dir = Path('models')
    required_files = ['lgb_model.txt', 'consolidation_config.json']
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print("Train models first with: python main.py")
        return False
    
    missing = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing model files: {', '.join(missing)}")
        print("Train models first with: python main.py")
        return False
    
    return True

def main():
    """Launch all components."""
    print("="*60)
    print("🚀 AML DASHBOARD LAUNCHER")
    print("="*60)
    print()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies installed")
    print()
    
    # Check models
    print("🧠 Checking trained models...")
    if not check_models():
        sys.exit(1)
    print("✅ Models found")
    print()
    
    # Get user preferences
    print("⚙️  Configuration:")
    simulator_rate = input("Transaction rate (default 2.0 tx/sec): ").strip() or "2.0"
    
    print()
    print("="*60)
    print("🎬 LAUNCHING COMPONENTS")
    print("="*60)
    print()
    
    processes = []
    
    try:
        # Start Flask API
        print("1️⃣  Starting Flask Inference API...")
        api_process = subprocess.Popen(
            [sys.executable, '-m', 'src.inference_api'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(('API', api_process))
        print("   ✅ API starting on http://localhost:5000")
        time.sleep(5)  # Wait for API to initialize
        
        # Start Transaction Simulator
        print("\n2️⃣  Starting Transaction Simulator...")
        simulator_process = subprocess.Popen(
            [sys.executable, 'transaction_simulator.py', '--rate', simulator_rate],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(('Simulator', simulator_process))
        print(f"   ✅ Simulator started ({simulator_rate} tx/sec)")
        time.sleep(2)
        
        # Start Streamlit Dashboard
        print("\n3️⃣  Starting Streamlit Dashboard...")
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(('Dashboard', dashboard_process))
        print("   ✅ Dashboard starting on http://localhost:8501")
        
        print()
        print("="*60)
        print("✅ ALL COMPONENTS RUNNING!")
        print("="*60)
        print()
        print("📊 Dashboard: http://localhost:8501")
        print("🔌 API: http://localhost:5000")
        print()
        print("Press Ctrl+C to stop all components...")
        print()
        
        # Keep running and show output
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  {name} process stopped unexpectedly!")
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping all components...")
        
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"   ✅ {name} stopped")
            except:
                proc.kill()
                print(f"   ⚠️  {name} force killed")
        
        print()
        print("="*60)
        print("👋 Shutdown complete!")
        print("="*60)

if __name__ == "__main__":
    main()
