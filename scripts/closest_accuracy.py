import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform

filelists = sys.argv[1] if len(sys.argv) >= 2 else 'filelists.txt'

with open(filelists) as f:
	labels = np.array([line.strip().split('\t')[1] for line in f], dtype=np.int16)

for features_name in sys.argv[2:]:
	features = np.fromfile(features_name, dtype=np.float32).reshape(labels.shape[0], -1)
	dist = squareform(pdist(features))
	np.fill_diagonal(dist, np.finfo(np.float32).max)
	closest = np.argmin(dist, axis=0)
	accuracy = sum([1 if labels[i] == labels[closest[i]] else 0 for i in range(labels.shape[0])]) / labels.shape[0]
	print('%s\t%f' % (features_name, accuracy))
