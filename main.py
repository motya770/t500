"""Entry point to launch the Economic Simulation dashboard.

Usage:
    python main.py          # Launches Streamlit app
    streamlit run app.py    # Alternative direct launch
"""

import os
import subprocess
import sys


def main():
    port = os.environ.get("PORT", "8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", port],
        check=True,
    )


if __name__ == "__main__":
    main()
