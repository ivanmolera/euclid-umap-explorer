from __future__ import annotations

import base64
import html

from .config import PROJECT_ROOT


def _icon_src(name: str) -> str:
    image = (PROJECT_ROOT / "assets" / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


SLED_ICON_SRC = _icon_src("sled_icon.png")
ALADIN_ICON_SRC = _icon_src("aladin_icon.png")
SLED_BUTTON_LABEL = f"![Check SLED]({SLED_ICON_SRC})"

ALADIN_LINK_CSS = """
a.aladin-icon-link:hover {
    filter: brightness(1.08);
}
a.aladin-icon-link:focus-visible {
    outline: 2px solid #93c5fd;
    outline-offset: 2px;
}
"""


def aladin_icon_link(url: str, *, css_class: str = "aladin-icon-link") -> str:
    classes = "aladin-icon-link" if css_class == "aladin-icon-link" else f"aladin-icon-link {css_class}"
    return (
        f'<a class="{html.escape(classes, quote=True)}" '
        f'href="{html.escape(url, quote=True)}" target="_blank" '
        'rel="noopener noreferrer" title="View in Aladin" aria-label="View in Aladin" '
        'style="display:inline-flex;align-items:center;justify-content:center;'
        'vertical-align:middle;flex:none;box-sizing:border-box;'
        'width:52px;height:34px;border-radius:4px;'
        'background:#e5e7eb;border:1px solid #cbd0d6">'
        f'<img src="{ALADIN_ICON_SRC}" alt="" width="44" height="26" '
        'style="object-fit:contain" /></a>'
    )
