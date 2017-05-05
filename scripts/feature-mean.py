import sys
import numpy as np
from PIL import Image

output = sys.argv[1] if len(sys.argv) >= 2 else 'features-raw.dat'

with open(output, 'wb') as f:
    for i in sys.argv[2:]:
        image = np.array(Image.open(i), dtype=np.uint32)
        image.mean(axis=0).mean(axis=0).astype(np.float32).tofile(f)
