import sys
import numpy as np
from sklearn.decomposition import PCA

n_components = int(sys.argv[1])
features_name = sys.argv[2] if len(sys.argv) >= 3 else 'features-raw.dat'
pca_name = sys.argv[3] if len(sys.argv) >= 4 else 'pca.dat'
output_name = sys.argv[4] if len(sys.argv) >= 5 else ''

features = np.fromfile(features_name, dtype=np.float32).reshape(-1, 4096)
ipca = PCA(n_components=n_components)
ipca.fit(features)
if (output_name):
	ipca.transform(features).astype(np.float32).tofile(output_name)
with open(pca_name, 'wb') as f:
	(-ipca.mean_).astype(np.float32).tofile(f)
	ipca.components_.astype(np.float32).tofile(f)
