import sys
import numpy as np
from PIL import Image

output = sys.argv[1] if len(sys.argv) >= 2 else 'features-raw.dat'

with open(output, 'wb') as f:
    for i in sys.argv[2:]:
        image = np.array(Image.open(i), dtype=np.uint32)
        np.histogram(image[:,:,0] * 256**2 + image[:,:,1] * 256 + image[:,:,2],
            4096, range=(0, 256**3))[0].astype(np.float32).tofile(f)
