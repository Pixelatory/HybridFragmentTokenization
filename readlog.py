import re

import numpy as np

results = []
with open('28-03-2023-09-34-58-log.txt', 'r') as f:
    idx = -1
    lines = f.readlines()
    for line in lines:
        if line.find('FOLD') != -1:
            continue
        elif re.match("^[+-]?(\d*\.)?\d+$", line):
            results[idx].append(float(line))
        else:
            results.append([])
            idx += 1
            results[idx].append(line.strip())

for i in range(len(results)):
    tmp = []
    print(results[i][0])
    for j in range(len(results)):
        if results[i][0] == results[j][0]:
            tmp.append(results[j][-1])
    if len(tmp) == 5:  # 5 is number of folds
        print(f"{round(float(np.mean(tmp)), 4):.4f}")

print(results)
exit(1)
lowest_test_loss_idx = np.argmin(results[-1][1:]) + 1
print(lowest_test_loss_idx)
for entry in results:
    print(entry[0], entry[lowest_test_loss_idx], max(entry[1:]), np.argmax(entry[1:]) + 1)
