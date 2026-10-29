import os
d = "research/michaelrustad_vqcc-dfthc-r10-merged-groups"
for root, dirs, files in os.walk(d):
    for f in files:
        print(os.path.join(root, f))
    if len(files) > 20:
        break
"