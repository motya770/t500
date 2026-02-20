"""Shared fixtures for Playwright end-to-end tests."""

import os
import signal
import subprocess
import time
import socket

import numpy as np
import pandas as pd
import pytest


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TEST_DATASET_NAME = "test_e2e_dataset"

# Indicators used in the test dataset (must match real World Bank codes)
TEST_INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
}

TEST_COUNTRIES = ["USA", "GBR", "DEU"]
TEST_YEARS = list(range(2018, 2024))


def _free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 30.0) -> None:
    """Block until the Streamlit server is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit server did not start within {timeout}s on port {port}")


@pytest.fixture(scope="session")
def test_dataset_path():
    """Create a small CSV dataset in data/ for tests that need saved data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{TEST_DATASET_NAME}.csv")

    np.random.seed(42)
    rows = []
    for country in TEST_COUNTRIES:
        for year in TEST_YEARS:
            rows.append({
                "country": country,
                "year": year,
                "NY.GDP.MKTP.CD": np.random.uniform(1e12, 5e12),
                "FP.CPI.TOTL.ZG": np.random.uniform(0, 10),
                "SL.UEM.TOTL.ZS": np.random.uniform(2, 15),
            })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    yield csv_path

    # Cleanup after all tests
    if os.path.exists(csv_path):
        os.remove(csv_path)


@pytest.fixture(scope="session")
def app_port():
    """Return a free port for the Streamlit server."""
    return _free_port()


@pytest.fixture(scope="session")
def streamlit_server(test_dataset_path, app_port):
    """Start a Streamlit server for the test session and tear it down afterwards."""
    app_py = os.path.join(os.path.dirname(__file__), "..", "..", "app.py")

    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    proc = subprocess.Popen(
        [
            "streamlit", "run", app_py,
            "--server.port", str(app_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    try:
        _wait_for_server(app_port)
    except TimeoutError:
        proc.kill()
        raise

    yield proc

    # Tear down: kill process group
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=10)


@pytest.fixture(scope="session")
def app_url(streamlit_server, app_port):
    """Base URL of the running Streamlit app."""
    return f"http://localhost:{app_port}"


@pytest.fixture()
def app_page(page, app_url):
    """Navigate to the app and wait for it to be fully loaded."""
    page.goto(app_url, wait_until="networkidle")
    # Wait for Streamlit to finish rendering
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=15000)
    # Also wait for the sidebar radio to appear (signals full render)
    page.wait_for_selector("[data-testid='stRadio']", timeout=10000)
    return page


# ---------------------------------------------------------------------------
# Streamlit interaction helpers
# ---------------------------------------------------------------------------


def click_sidebar_nav(page, label_substring: str) -> None:
    """Click a sidebar radio option by matching text.

    Targets the radio widget inside the sidebar specifically, to avoid
    hitting other elements (e.g. banner/footer) that contain similar text.
    """
    radio = page.locator("[data-testid='stRadio']")
    radio.get_by_text(label_substring, exact=False).first.click()
    # Wait for Streamlit to re-render
    page.wait_for_timeout(2000)
    page.wait_for_load_state("networkidle")


def select_radio_option(page, option_text: str) -> None:
    """Select a radio option in the main area by label text."""
    page.get_by_text(option_text, exact=True).click()
    page.wait_for_timeout(1000)


def select_selectbox_option(page, selectbox_label: str, option_text: str) -> None:
    """Open a Streamlit selectbox and pick an option.

    ``selectbox_label`` is the visible label above the widget.
    ``option_text`` is the option to choose from the dropdown.
    """
    # Find the selectbox container by its label
    label_el = page.get_by_text(selectbox_label, exact=True)
    # The selectbox widget is typically the next sibling container
    container = label_el.locator("xpath=ancestor::div[@data-testid='stSelectbox'] | ancestor::div[contains(@class,'stSelectbox')]")
    # If we can't find it by ancestor, try a broader approach
    try:
        container.first.click()
    except Exception:
        # Fallback: find the selectbox that's near the label
        page.locator(f"[data-testid='stSelectbox']:has-text('{selectbox_label}')").first.click()

    page.wait_for_timeout(300)

    # Click the option in the dropdown list
    page.get_by_role("option", name=option_text).click()
    page.wait_for_timeout(1000)
