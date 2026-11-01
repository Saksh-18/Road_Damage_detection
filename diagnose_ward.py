"""Explain why a coordinate maps to a given ward.

Usage:  python diagnose_ward.py [lat] [lon]
"""
import sys

import geo_utils as geo

lat = float(sys.argv[1]) if len(sys.argv) > 1 else 19.066929
lon = float(sys.argv[2]) if len(sys.argv) > 2 else 72.826070

wards = geo.load_wards()
print(f"Point: {lat}, {lon}   |   wards loaded: {len(wards)}\n")
print(f"{'ward':6} {'covers':7} {'dist_m':>7} {'area_km2':>9}  repaired")
for r in geo.diagnose_point(lat, lon, wards)[:6]:
    print(f"{r['ward']:6} {str(r['covers']):7} {r['distance_m']:>7} {r['area_km2']:>9}  {r['was_invalid']}")

found = geo.find_ward(lat, lon, wards)
print("\nfind_ward  ->", found)
print("CSV row    ->", geo.ward_contact(found))