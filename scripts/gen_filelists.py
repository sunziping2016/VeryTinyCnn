import sys
import os

image_folder = sys.argv[1] if len(sys.argv) >= 2 else 'image'
output = sys.argv[2] if len(sys.argv) >= 3 else 'filelists.txt'
classes = set()
images = []
for i in os.listdir(image_folder):
    c, _ = i.split('_')
    n, e = _.split('.')
    newi = '%s_%05d.%s' % (c, int(n), e)
    if i != newi:
        os.rename(os.path.join(image_folder, i), os.path.join(image_folder, newi))
    classes.add(c)
    images.append((c, os.path.join(image_folder, newi)))
classes = {v: i  for i, v in enumerate(sorted(classes))}
with open(output, 'w') as f:
    for i in sorted(images):
        f.write('%s\t%d\n' %(i[1], classes[i[0]]))
