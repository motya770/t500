"""Tests for ui/theme.py module."""

import pytest


class TestThemeConstants:
    """Tests for theme color constants and templates."""

    def test_color_constants_are_strings(self):
        from ui.theme import (
            COAL, DARK_IRON, IRON, STEEL, MAHOGANY, DARK_WOOD,
            WOOD, BRASS, COPPER, BRONZE, GOLD, EMBER, FIRE_ORANGE,
            CREAM, PARCHMENT, STEAM, SMOKE,
        )
        colors = [
            COAL, DARK_IRON, IRON, STEEL, MAHOGANY, DARK_WOOD,
            WOOD, BRASS, COPPER, BRONZE, GOLD, EMBER, FIRE_ORANGE,
            CREAM, PARCHMENT, STEAM, SMOKE,
        ]
        for color in colors:
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format

    def test_chart_colors_is_list(self):
        from ui.theme import CHART_COLORS
        assert isinstance(CHART_COLORS, list)
        assert len(CHART_COLORS) > 0
        for color in CHART_COLORS:
            assert isinstance(color, str)

    def test_heatmap_scale_is_list(self):
        from ui.theme import HEATMAP_SCALE
        assert isinstance(HEATMAP_SCALE, list)
        assert len(HEATMAP_SCALE) > 0

    def test_diverging_scale_is_list(self):
        from ui.theme import DIVERGING_SCALE
        assert isinstance(DIVERGING_SCALE, list)
        assert len(DIVERGING_SCALE) > 0

    def test_css_is_nonempty_string(self):
        from ui.theme import STEAM_CSS
        assert isinstance(STEAM_CSS, str)
        assert len(STEAM_CSS) > 100

    def test_header_banner_is_html(self):
        from ui.theme import HEADER_BANNER
        assert isinstance(HEADER_BANNER, str)
        assert "<" in HEADER_BANNER

    def test_sidebar_header_is_html(self):
        from ui.theme import SIDEBAR_HEADER
        assert isinstance(SIDEBAR_HEADER, str)
        assert "<" in SIDEBAR_HEADER

    def test_sidebar_footer_is_html(self):
        from ui.theme import SIDEBAR_FOOTER
        assert isinstance(SIDEBAR_FOOTER, str)
        assert "<" in SIDEBAR_FOOTER


class TestApplySteamStyle:
    """Tests for apply_steam_style function."""

    def test_returns_figure(self):
        import plotly.graph_objects as go
        from ui.theme import apply_steam_style

        fig = go.Figure()
        result = apply_steam_style(fig)
        assert result is fig  # Returns same object

    def test_applies_template(self):
        import plotly.graph_objects as go
        from ui.theme import apply_steam_style

        fig = go.Figure()
        apply_steam_style(fig)
        assert fig.layout.template is not None


class TestSteamTemplate:
    """Tests for the Plotly template registration."""

    def test_template_registered(self):
        import plotly.io as pio
        # Importing theme registers the template
        import ui.theme  # noqa: F401
        assert "steam_train" in pio.templates
