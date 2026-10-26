from __future__ import annotations

import hashlib
import html

import streamlit as st

from .config import ENABLE_SLED_CHECK
from .sled import SledClient, SledQuery, SledResult, credentials_available
from .service_icons import SLED_BUTTON_LABEL


@st.cache_resource(show_spinner=False)
def sled_client() -> SledClient:
    return SledClient()


def result_presentation(result: SledResult | None) -> tuple[str, str, str]:
    if result is None:
        return "Not checked", "search", "Check public SLED records near this object's coordinates."
    if result.status == "pending":
        return "Checking...", "hourglass_top", "SLED is being checked in the background. You can continue using ESCOPE."
    if result.status == "no_match":
        return "No public match", "search_off", "No public SLED entry was found within the search radius. This does not rule out lensing or establish a new discovery."
    if result.status == "matched":
        if len(result.matches) > 1:
            return f"{len(result.matches)} nearby entries", "difference", "Several public SLED entries are nearby. Review the positions before identifying the same system."
        match = result.matches[0]
        label = {
            "CONFIRMED": "Confirmed lens nearby", "CANDIDATE": "Candidate nearby",
            "CONTAMINANT": "Contaminant nearby", "UNSPECIFIED": "Entry nearby",
        }[match.flag]
        icon = {"CONFIRMED": "verified", "CONTAMINANT": "report"}.get(match.flag, "info")
        return label, icon, (
            f"{match.name}: SLED status {match.flag}; separation {match.separation_arcsec:.2f} arcsec. "
            "A positional association must be checked; ESCOPE's original label is unchanged."
        )
    return (
        {"timeout": "Timed out", "error": "Check failed", "unavailable": "Unavailable", "busy": "Busy"}.get(result.status, "Check failed"),
        "schedule" if result.status == "timeout" else "warning",
        result.message,
    )


@st.fragment(run_every=1)
def render_sled_check(object_id: str, ra: float | None, dec: float | None, *, key: str) -> None:
    """Refresh only lightweight worker/cache state, never network or analysis."""
    if not ENABLE_SLED_CHECK:
        return
    try:
        query = SledQuery(float(ra), float(dec))
    except (TypeError, ValueError):
        query = None
    client = sled_client()
    identity = (str(object_id), query)
    result = client.get(query) if query else None
    configured = credentials_available()
    label, _, tooltip = result_presentation(result)
    if not configured:
        label, tooltip = "Unavailable", "SLED credentials are not configured on the server."
    elif query is None:
        label, tooltip = "Unavailable", "Valid sky coordinates are not available for this object."
    if query:
        tooltip += f" Search radius: {query.radius_arcsec:g} arcsec."
    pending = result is not None and result.status == "pending"
    button_help = "Check SLED. " + tooltip
    if configured and query is not None and not pending:
        button_help += " Click to check or refresh. Results are cached; the check runs in the background."
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    container_key = f"{key}_sled_panel_{digest}"
    st.html(f"""<style>
        .st-key-{container_key} button img[alt="Check SLED"] {{
            height: 24px; width: 24px; max-height: none; object-fit: contain;
        }}
    </style>""")
    with st.container(key=container_key, horizontal=True, vertical_alignment="center", gap="small"):
        clicked = st.button(
            SLED_BUTTON_LABEL, type="tertiary",
            key=f"{key}_sled_{digest}", help=button_help,
            disabled=not configured or query is None or pending,
        )
        if clicked:
            result = client.request(query, refresh=result is not None)
            label, _, tooltip = result_presentation(result)
        st.markdown(
            f'<span title="{html.escape(tooltip, quote=True)}" '
            f'style="font-size:0.85rem">{html.escape(label)}</span>',
            unsafe_allow_html=True,
        )
        if result and result.checked_at:
            with st.popover("Details", icon=":material/info:", type="tertiary"):
                st.caption(f"Checked: {result.checked_at} (UTC)")
                if query:
                    st.caption(f"Public records within {query.radius_arcsec:g} arcsec. Original A/B/C labels are unchanged.")
                if result.matches:
                    st.dataframe([
                        {"Name": m.name, "SLED status": m.flag,
                         "Separation (arcsec)": round(m.separation_arcsec, 3),
                         "RA (deg)": m.ra, "Dec (deg)": m.dec}
                        for m in result.matches
                    ], hide_index=True)
                    papers = [
                        {"System": m.name, "Year": p.year, "Author": p.author,
                         "Title": p.title, "Reference": p.reference}
                        for m in result.matches for p in m.papers
                    ]
                    if papers:
                        st.dataframe(papers, hide_index=True)
                    st.caption("Nearby entries are possible associations, not automatic identifications.")
                elif result.status == "no_match":
                    st.write("No public entry was found. This does not rule out lensing or establish a new discovery.")
                else:
                    st.write(result.message)
                if query:
                    st.link_button("Open SLED search", query.search_url(), icon=":material/open_in_new:")
