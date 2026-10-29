"""Write the SomeJoint fused-DQI notebook's embedded SOURCES to disk and check their SHA-256."""
import hashlib, pathlib, json, sys
cell = pathlib.Path(sys.argv[1]).read_text().split("\n")
root = pathlib.Path(sys.argv[2])
ns = {}
exec("\n".join(cell[:10]), ns)
(root / "data").mkdir(parents=True, exist_ok=True)
bad = 0
for name, body in ns["SOURCES"].items():
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    if hashlib.sha256(p.read_bytes()).hexdigest() != ns["SOURCE_HASHES"][name]:
        bad += 1
        print("hash mismatch", name)
(root / "members.json").write_text(json.dumps(ns["MEMBERS"]))
print(len(ns["SOURCES"]), "files written, bad", bad)
print(ns["BUILDS"], list(ns["COMPETITION_FILES"].keys()))
for name in sorted(ns["SOURCES"]):
    print(f"{len(ns['SOURCES'][name]):8d} {name}")
