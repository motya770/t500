"""Entry point to launch the Economic Simulation dashboard.

Usage:
    python main.py          # Launches Streamlit app
    streamlit run app.py    # Alternative direct launch
"""

import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)


if __name__ == "__main__":
    main()
