ids = []
with open("./mappings") as f:
    lines = f.readlines()
    lines = [l.strip().split() for l in lines[1:]]
    ids = set([int(idx) for idx, name in lines])
    # print(lines[:20])
print(ids)
missing = []
for i in range(463):
    if i not in ids:
        missing.append(i)
print(missing)
    