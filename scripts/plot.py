import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

points_name = sys.argv[1] if len(sys.argv) >= 2 else 'points.dat'
output_name = sys.argv[2] if len(sys.argv) >= 3 else 'tsne.png'
filelists = sys.argv[3] if len(sys.argv) >= 4 else 'filelists.txt'
labels_name = sys.argv[4] if len(sys.argv) >= 5 else ''

cmap = ListedColormap(['pink', 'red', 'orange', 'yellow', 'brown', 'green','cyan', 'blue', 'purple', 'black'])

with open(filelists) as f:
	labels = np.array([line.strip().split('\t')[1] for line in f], dtype=np.int16)

points = np.fromfile(points_name, dtype=np.float32).reshape(-1, 2)

fig = plt.figure(figsize=(10, 10), tight_layout=True)
subplt = fig.add_subplot(1, 1, 1)
subplt.scatter(points[:, 0], points[:, 1], s=5, c=labels, cmap=cmap, edgecolors='none', marker='o')
subplt.axis('off')
if labels_name:
	with open(labels_name) as f:
		labels_map = f.readlines()
	colors = np.linspace(0, 1, len(labels_map))
	plt.legend(handles=[Patch(color=cmap(colors[i]), label=v) for i, v in enumerate(labels_map)], loc=4)
fig.savefig(output_name)
