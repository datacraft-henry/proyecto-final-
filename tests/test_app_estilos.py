import re

import pytest

from app.estilos import DARK, LIGHT, PALETTES, build_css


def _keys(palette: dict, prefix: str = "") -> set[str]:
    """Claves anidadas ('level.alto', 'streamlit.base'), para comparar paletas."""
    keys = set()
    for key, value in palette.items():
        keys |= _keys(value, f"{prefix}{key}.") if isinstance(value, dict) else {f"{prefix}{key}"}
    return keys


def test_both_palettes_define_the_same_tokens():
    assert _keys(LIGHT) == _keys(DARK)
    assert set(PALETTES) == {"light", "dark"}
    assert LIGHT["streamlit"]["base"] == "light" and DARK["streamlit"]["base"] == "dark"


@pytest.mark.parametrize("palette", [LIGHT, DARK], ids=["light", "dark"])
def test_css_is_built_from_the_palette_without_leftover_placeholders(palette):
    css = build_css(palette)
    assert css.strip().startswith("<style>") and css.strip().endswith("</style>")
    assert palette["bg"] in css and palette["surface"] in css and palette["level"]["alto"] in css
    assert not re.search(r"\{p\[|\{lvl\[|\{soft\[|\{ink\[", css)  # ningún token quedó sin resolver
