"""Convenience entry point for this project.

Train the model with:
    .\\venv\\Scripts\\python.exe train_model.py

Launch the web dashboard with:
    .\\venv\\Scripts\\python.exe -m streamlit run dashboard.py
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
dashboard = ROOT / "dashboard.py"

if __name__ == "__main__":
    if not dashboard.exists():
        raise FileNotFoundError(f"Dashboard not found: {dashboard}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard)],
        cwd=str(ROOT),
        check=False,
    )
