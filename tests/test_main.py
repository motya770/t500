"""Tests for main.py module."""

import pytest
from unittest.mock import patch, MagicMock


class TestMain:
    """Tests for the main() entry point."""

    @patch("main.subprocess.run")
    def test_main_calls_streamlit(self, mock_run):
        """main() launches streamlit run app.py."""
        from main import main
        main()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "streamlit" in args
        assert "run" in args
        assert "app.py" in args

    @patch("main.subprocess.run")
    def test_main_uses_check_true(self, mock_run):
        """main() passes check=True to subprocess.run."""
        from main import main
        main()

        mock_run.assert_called_once()
        assert mock_run.call_args[1].get("check") is True or \
               (len(mock_run.call_args) > 1 and mock_run.call_args[1].get("check") is True)

    @patch("main.subprocess.run")
    def test_main_uses_sys_executable(self, mock_run):
        """main() uses sys.executable as the Python interpreter."""
        import sys
        from main import main
        main()

        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable

    @patch("main.subprocess.run", side_effect=Exception("Streamlit failed"))
    def test_main_propagates_errors(self, mock_run):
        """main() does not swallow subprocess errors."""
        from main import main
        with pytest.raises(Exception, match="Streamlit failed"):
            main()
