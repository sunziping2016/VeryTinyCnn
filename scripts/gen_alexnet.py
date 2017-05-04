import sys
import torch.utils.model_zoo as model_zoo
state_dict = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
# Remove final classification layer
del state_dict['classifier.6.weight']
del state_dict['classifier.6.bias']
filename = sys.argv[1] if len(sys.argv) >= 2 else 'alexnet.dat'
with open(filename, 'wb') as f:
	for t in state_dict.values():
		t.data.view(-1).numpy().tofile(f)
