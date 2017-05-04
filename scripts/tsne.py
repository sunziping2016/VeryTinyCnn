import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors

features_name = sys.argv[1] if len(sys.argv) >= 2 else 'features.dat'
filelists = sys.argv[2] if len(sys.argv) >= 3 else 'filelists.txt'
output_name = sys.argv[3] if len(sys.argv) >= 4 else 'tsne.png'

with open(filelists) as f:
	labels = np.array([line.strip().split('\t')[1] for line in f], dtype=np.int16)
features = np.fromfile(features_name, dtype=np.float32).reshape(labels.shape[0], -1)
if features.shape[1] > 2:
	model = TSNE(n_components=2, verbose=True)
	points = model.fit_transform(features)
elif features.shape[1] == 2:
	points = features
elif features.shape[1] == 1:
	points = np.concatenate([features, features], axis=1)
fig = plt.figure(figsize=(10, 10), tight_layout=True)
subplt = fig.add_subplot(1, 1, 1)
subplt.scatter(points[:, 0], points[:, 1], s=5, c=labels, cmap='plasma', edgecolors='none', marker='o')
subplt.axis('off')
fig.savefig(output_name)
