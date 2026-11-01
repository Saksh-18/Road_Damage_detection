"""Location helpers for the road damage app.

Photo GPS (EXIF)  ->  BMC ward  ->  locality/road  ->  official contact row.

Install (in the environment that runs your app):
    pip install shapely requests pillow

Data files you provide, saved next to this file:
  mumbai_wards.geojson  Polygons of BMC administrative wards. Check that it is the
                        CURRENT boundary set; older open files may use older wards.
  ward_contacts.csv     Columns: ward,office,phone,email,source_url,last_verified
                        Fill it ONLY from official BMC pages and record the source.

IMPORTANT: set USER_AGENT below (or the GEO_USER_AGENT environment variable) to a
name for your project plus a real contact email. OpenStreetMap's Nominatim blocks
placeholder or generic user agents with HTTP 403.
"""
import csv
import io
import json
import math
import os
import time
from pathlib import Path

import requests
from PIL import Image
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
GPS_IFD = 0x8825  # EXIF pointer to the GPS block

USER_AGENT = os.environ.get(
    "GEO_USER_AGENT", "road-damage-college-project/1.0 (contact: jhasakshi3089@gmail.com)"
)
CACHE_PATH = HERE / "geocode_cache.json"
NEAREST_WARD_MAX_M = 300  # tolerance for points that fall just outside a coarse boundary


# ---------- 1. GPS from the photo ----------
def _to_degrees(dms, ref):
    d, m, s = (float(v) for v in dms)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def extract_gps(image_bytes):
    """Return (lat, lon) from the photo's EXIF, or None if it has no location tag.

    WhatsApp, Telegram and screenshots strip this data, so use original camera files.
    """
    try:
        gps = Image.open(io.BytesIO(image_bytes)).getexif().get_ifd(GPS_IFD)
        if not gps or 2 not in gps or 4 not in gps:
            return None
        lat = _to_degrees(gps[2], gps.get(1, "N"))
        lon = _to_degrees(gps[4], gps.get(3, "E"))
        return (lat, lon) if (lat or lon) else None
    except Exception:
        return None


# ---------- 2. Ward lookup (point in polygon) ----------
_NAME_KEYS = ("ward", "WARD", "Ward", "ward_name", "WARD_NAME", "name", "NAME", "Name")
_WAS_INVALID = set()  # ward names whose raw polygon was invalid and got repaired


def _repair(geom):
    """Fix self-intersecting / invalid polygons, which can give wrong containment answers."""
    if geom.is_valid:
        return geom, False
    try:
        fixed = make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    if fixed.geom_type == "GeometryCollection":
        polys = [g for g in fixed.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        fixed = unary_union(polys) if polys else geom.buffer(0)
    return fixed, True


def load_wards(path=HERE / "mumbai_wards.geojson"):
    """Load ward polygons as a list of (ward_name, shapely_geometry)."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    wards = []
    _WAS_INVALID.clear()
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        name = next((str(props[k]) for k in _NAME_KEYS if props.get(k)), None)
        if name is None:  # unknown property name: fall back to the first value
            name = str(next(iter(props.values()), "unknown"))
        geom, was_invalid = _repair(shape(feat["geometry"]))
        if was_invalid:
            _WAS_INVALID.add(name)
        wards.append((name, geom))
    return wards


def _meters(deg):
    return deg * 111_000  # rough: one degree is about 111 km


def find_ward(lat, lon, wards, max_snap_m=NEAREST_WARD_MAX_M):
    """Return the ward containing the point.

    - If several polygons contain it (overlaps in the data), the smallest wins.
    - If none does, the nearest ward within max_snap_m is used (coarse coastlines).
    - Otherwise None.
    """
    pt = Point(lon, lat)  # GeoJSON order is (lon, lat)
    hits = [(name, g) for name, g in wards if g.covers(pt)]
    if hits:
        return min(hits, key=lambda x: x[1].area)[0]
    nearest = min(((name, g.distance(pt)) for name, g in wards), key=lambda x: x[1], default=None)
    if nearest and _meters(nearest[1]) <= max_snap_m:
        return nearest[0]
    return None


def diagnose_point(lat, lon, wards):
    """Rows describing how every ward relates to a point (for debugging bad lookups)."""
    pt = Point(lon, lat)
    rows = []
    for name, g in wards:
        rows.append(
            {
                "ward": name,
                "covers": g.covers(pt),
                "distance_m": round(_meters(g.distance(pt))),
                "area_km2": round(g.area * 111 * 111 * math.cos(math.radians(lat)), 1),
                "was_invalid": name in _WAS_INVALID,
            }
        )
    return sorted(rows, key=lambda r: (not r["covers"], r["distance_m"]))


# ---------- 3. Locality / road name ----------
_last_call = 0.0


def _cache_key(lat, lon):
    return f"{lat:.4f},{lon:.4f}"  # about 11 m resolution


def _read_cache():
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(cache):
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def reverse_geocode(lat, lon, user_agent=None):
    """Look up road and locality via OpenStreetMap's Nominatim.

    The public server requires a real, identifying User-Agent with a contact, allows
    at most 1 request per second, and expects results to be cached. Good results are
    cached in geocode_cache.json, so repeated demo locations never hit the server.
    """
    global _last_call
    ua = user_agent or USER_AGENT
    key = _cache_key(lat, lon)
    cache = _read_cache()
    if key in cache:
        return cache[key]
    if "YOUR_EMAIL_HERE" in ua or "example." in ua.lower():
        return {"error": "Set USER_AGENT in geo_utils.py to your real contact email (OpenStreetMap blocks placeholders)."}

    wait = 1.1 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18, "addressdetails": 1},
            headers={"User-Agent": ua, "Accept-Language": "en"},
            timeout=10,
        )
        if r.status_code == 403:
            return {"error": "HTTP 403: blocked by OpenStreetMap. Use a real contact in USER_AGENT; "
                             "if you already did, your IP may be temporarily blocked, so try again later."}
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # report WHY it failed instead of failing silently
        return {"error": f"{type(e).__name__}: {str(e)[:110]}"}
    if j.get("error"):
        return {"error": str(j["error"])[:110]}
    a = j.get("address", {})
    result = {
        "road": a.get("road") or a.get("pedestrian") or a.get("footway"),
        "locality": a.get("neighbourhood") or a.get("suburb") or a.get("quarter"),
        "suburb": a.get("suburb"),
        "city": a.get("city") or a.get("town") or a.get("state_district"),
        "postcode": a.get("postcode"),
        "display": j.get("display_name"),
    }
    cache[key] = result
    _write_cache(cache)
    return result


# ---------- 4. Official contact for the ward ----------
def _norm(ward):
    return str(ward).upper().replace("WARD", "").strip()


def ward_contact(ward, path=HERE / "ward_contacts.csv"):
    """Return the CSV row (dict) for this ward, or None if missing."""
    path = Path(path)
    if not ward or not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if _norm(row.get("ward", "")) == _norm(ward):
                return row
    return None