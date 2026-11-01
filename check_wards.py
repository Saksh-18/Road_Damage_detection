import json
import geo_utils as geo

d = json.load(open("mumbai_wards.geojson", encoding="utf-8"))
print("wards:", len(d["features"]))
print("first properties:", d["features"][0]["properties"])

wards = geo.load_wards()
print("ward names:", sorted(n for n, _ in wards))
print("test point:", geo.find_ward(19.06689091267172, 72.82612466441749, wards))  # replace with a spot you know