import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
print(not (a - b).round(4).any())
