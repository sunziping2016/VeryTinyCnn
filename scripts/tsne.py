import sys
import numpy as np
from sklearn.manifold import TSNE


features_num = int(sys.argv[1])
features_name = sys.argv[2] if len(sys.argv) >= 2 else 'features.dat'
output_name = sys.argv[3] if len(sys.argv) >= 4 else 'tsne.dat'

features = np.fromfile(features_name, dtype=np.float32).reshape(-1, features_num)
if features.shape[1] > 2:
	model = TSNE(n_components=2, verbose=True)
	points = model.fit_transform(features)
elif features.shape[1] == 2:
	points = features
elif features.shape[1] == 1:
	points = np.concatenate([features, features], axis=1)
points.astype(dtype=np.float32).tofile(output_name)
