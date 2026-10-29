import json
import os
d = "research/somejoint_qcc-our-chemistry-b154-c62"
for f in os.listdir(d):
    print(f)
    if f.endswith(".json"):
        data = json.load(open(os.path.join(d, f)))
        print("  Keys:", list(data.keys())[:10])
        for key in ["score", "status", "cost", "lambda_eff"]:
            if key in data:
                print("  " + key + ":", data[key])