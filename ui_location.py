"""Polished "Location & Responsible Authority" panel for the Streamlit app (dark theme).

In appv8.py, delete your current st.map / st.write location block and call:

    import ui_location                       # with your other imports
    ...
    ui_location.render_location_panel(lat, lon, ward, addr, contact)

  lat, lon  decimal degrees        ward     ward label or None
  addr      dict from geo.reverse_geocode (may be {} or contain "error")
  contact   row from geo.ward_contact (dict or None)
  summary   optional text used as the body of the "Email ward office" button
"""
import html
import re
from urllib.parse import quote

PORTAL_URL = "https://portal.mcgm.gov.in"  # BMC online services / complaints portal

_ICON_PATHS = {
    "pin": '<path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "building": '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4M8 6h.01M12 6h.01M16 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01"/>',
    "phone": '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z"/>',
    "mail": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>',
    "flag": '<path d="M4 22V4M4 4h13l-2 4 2 4H4"/>',
    "map": '<path d="M9 3 3 5v16l6-2 6 2 6-2V3l-6 2-6-2zM9 3v16M15 5v16"/>',
}

CSS = """
.rdl-head{display:flex;align-items:center;gap:12px;margin:6px 0 16px}
.rdl-badge{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#4f8cff,#7c5cff);color:#fff;flex:none}
.rdl-title{font-size:1.15rem;font-weight:700;letter-spacing:.2px;color:#e8eefc}
.rdl-sub{font-size:.82rem;color:#8d9ab8;margin-top:2px}
.rdl-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:14px;padding:14px 16px;margin-bottom:12px}
.rdl-card-h{display:flex;align-items:center;gap:8px;font-size:.72rem;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:#8d9ab8;margin-bottom:8px}
.rdl-row{display:flex;justify-content:space-between;gap:16px;padding:8px 0;border-top:1px solid rgba(255,255,255,.06)}
.rdl-card-h + .rdl-row{border-top:0}
.rdl-k{color:#8d9ab8;font-size:.85rem;flex:none}
.rdl-v{color:#e8eefc;font-size:.9rem;font-weight:500;text-align:right;word-break:break-word}
.rdl-muted{color:#6b7794;font-weight:400}
.rdl-mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85rem}
.rdl-ward{display:flex;align-items:center;gap:14px}
.rdl-chip{min-width:58px;height:58px;padding:0 12px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.35rem;font-weight:800;color:#fff;background:linear-gradient(135deg,#4f8cff,#7c5cff);box-shadow:0 6px 18px rgba(79,140,255,.28)}
.rdl-chip.off{background:rgba(251,191,36,.14);color:#fbbf24;box-shadow:none;font-size:1.1rem}
.rdl-ward-t{font-weight:600;color:#e8eefc}
.rdl-ward-s{font-size:.8rem;color:#8d9ab8;margin-top:2px}
.rdl-empty{border:1px dashed rgba(255,255,255,.18);border-radius:10px;padding:12px;font-size:.85rem;color:#8d9ab8;line-height:1.45}
.rdl-actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:2px}
.rdl-btn{display:inline-flex;align-items:center;gap:8px;padding:9px 14px;border-radius:10px;font-size:.85rem;font-weight:600;text-decoration:none !important;border:1px solid rgba(255,255,255,.16);color:#e8eefc !important;background:rgba(255,255,255,.05)}
.rdl-btn:hover{background:rgba(255,255,255,.11)}
.rdl-btn.primary{background:linear-gradient(135deg,#4f8cff,#7c5cff);border-color:transparent;color:#fff !important}
.rdl-link{color:#7fb0ff !important;text-decoration:none !important}
.rdl-foot{font-size:.75rem;color:#6b7794;margin-top:10px;line-height:1.45}
.rdl-hint{font-size:.74rem;color:#fbbf24;margin-top:6px;line-height:1.4}
.rdl svg{fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.rdl-card-h svg{width:15px;height:15px}
.rdl-badge svg{width:20px;height:20px}
.rdl-btn svg{width:16px;height:16px}
"""


# ---------- small helpers ----------
def _flat(s):
    """Remove newlines/indentation so Markdown never treats the HTML as a code block."""
    return "".join(line.strip() for line in s.splitlines())


def _esc(v):
    return html.escape(str(v), quote=True)


def _clean(v):
    """Return a stripped string, or '' for None / blank / 'None'."""
    v = "" if v is None else str(v).strip()
    return "" if v.lower() == "none" else v


def _val(v, mono=False):
    v = _clean(v)
    cls = "rdl-v rdl-mono" if mono else "rdl-v"
    return f'<span class="{cls}">{_esc(v)}</span>' if v else '<span class="rdl-v rdl-muted">Not available</span>'


def _icon(name):
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{_ICON_PATHS[name]}</svg>'


def _row(key, value_html):
    return f'<div class="rdl-row"><span class="rdl-k">{_esc(key)}</span>{value_html}</div>'


def _card(icon, title, body):
    return f'<div class="rdl-card"><div class="rdl-card-h">{_icon(icon)}{_esc(title)}</div>{body}</div>'


def _btn(href, icon, label, primary=False, external=True):
    cls = "rdl-btn primary" if primary else "rdl-btn"
    target = ' target="_blank" rel="noopener noreferrer"' if external else ""
    return f'<a class="{cls}" href="{_esc(href)}"{target}>{_icon(icon)}{_esc(label)}</a>'


def _safe_http(url):
    url = _clean(url)
    return url if url.lower().startswith(("http://", "https://")) else ""


# ---------- HTML builders (pure functions) ----------
def build_header_html():
    return _flat(
        f'<style>{CSS}</style><div class="rdl"><div class="rdl-head">'
        f'<div class="rdl-badge">{_icon("pin")}</div>'
        '<div><div class="rdl-title">Location &amp; Responsible Authority</div>'
        '<div class="rdl-sub">Where the damage was found and who to contact</div></div></div></div>'
    )


def build_cards_html(lat, lon, ward, addr=None, contact=None, summary=None, portal_url=PORTAL_URL):
    addr = addr or {}
    contact = contact or {}
    lat, lon = float(lat), float(lon)
    ward = _clean(ward)
    maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

    # --- Location card ---
    coords = f"{abs(lat):.6f}\u00b0{'N' if lat >= 0 else 'S'}, {abs(lon):.6f}\u00b0{'E' if lon >= 0 else 'W'}"
    locality = _clean(addr.get("locality")) or _clean(addr.get("suburb"))
    loc_body = (
        _row("Coordinates", _val(coords, mono=True))
        + _row("Road", _val(addr.get("road")))
        + _row("Locality", _val(locality))
        + _row("City", _val(addr.get("city")))
    )
    err = _clean(addr.get("error"))
    if err:
        loc_body += f'<div class="rdl-hint">Address lookup failed ({_esc(err)}). The ward below is still valid.</div>'
    cards = _card("pin", "Location", loc_body)

    # --- Jurisdiction card ---
    if ward:
        ward_body = (
            f'<div class="rdl-ward"><div class="rdl-chip">{_esc(ward)}</div><div>'
            f'<div class="rdl-ward-t">{_esc(ward)} Ward</div>'
            '<div class="rdl-ward-s">BMC administrative ward</div></div></div>'
        )
    else:
        ward_body = (
            '<div class="rdl-ward"><div class="rdl-chip off">N/A</div><div>'
            '<div class="rdl-ward-t">Outside mapped wards</div>'
            '<div class="rdl-ward-s">This point is not inside any ward boundary.</div></div></div>'
        )
    cards += _card("shield", "Jurisdiction", ward_body)

    # --- Contact card ---
    office, phone, email = (_clean(contact.get(k)) for k in ("office", "phone", "email"))
    if office or phone or email:
        body = _row("Office", _val(office)) + _row("Phone", _val(phone)) + _row("Email", _val(email))
        src = _safe_http(contact.get("source_url"))
        verified = _clean(contact.get("last_verified"))
        bits = []
        if src:
            bits.append(f'Source: <a class="rdl-link" href="{_esc(src)}" target="_blank" rel="noopener noreferrer">official BMC page</a>')
        if verified:
            bits.append(f"Verified: {_esc(verified)}")
        if bits:
            body += f'<div class="rdl-foot">{" &middot; ".join(bits)}</div>'
    else:
        who = f"ward {_esc(ward)}" if ward else "this location"
        body = (
            f'<div class="rdl-empty">Contact details for {who} have not been added yet. '
            "Use the BMC portal below to file the complaint.</div>"
        )
    cards += _card("building", "Ward office (BMC)", body)

    # --- Action buttons ---
    buttons = [_btn(portal_url, "flag", "File a complaint", primary=True)]
    first_phone = re.split(r"[,/;]", phone)[0] if phone else ""
    digits = re.sub(r"[^\d+]", "", first_phone)
    if len(digits) >= 5:
        buttons.append(_btn(f"tel:{digits}", "phone", "Call ward office", external=False))
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        body_text = summary or (
            f"Road damage reported at {lat:.6f}, {lon:.6f}\n"
            f"Google Maps: {maps_url}\n"
            f"Ward: {ward or 'unknown'}\n\nPlease inspect and repair."
        )
        subject = f"Road damage report - Ward {ward}" if ward else "Road damage report"
        buttons.append(_btn(f"mailto:{email}?subject={quote(subject)}&body={quote(body_text)}", "mail", "Email ward office", external=False))
    buttons.append(_btn(maps_url, "map", "Open in Google Maps"))
    cards += f'<div class="rdl-actions">{"".join(buttons)}</div>'
    cards += (
        '<div class="rdl-foot">Reference information only. Confirm details on the official '
        "BMC website before filing.</div>"
    )
    return _flat(f'<div class="rdl">{cards}</div>')


# ---------- Streamlit entry point ----------
def render_location_panel(lat, lon, ward, addr=None, contact=None, summary=None):
    import pandas as pd
    import streamlit as st

    st.markdown(build_header_html(), unsafe_allow_html=True)
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        df = pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]})
        try:
            st.map(df, size=12, zoom=15, height=600)
        except TypeError:  # older Streamlit without these options
            st.map(df)
    with right:
        st.markdown(build_cards_html(lat, lon, ward, addr, contact, summary), unsafe_allow_html=True)