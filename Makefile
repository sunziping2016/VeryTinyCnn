features = 4 8 12 16 20 40 80 200

CXXFLAGS = -Iinclude -std=c++11 -O3 -Wall -Wextra -Wno-unused-parameter -lpthread -lX11
AVX_ENABLED = $(shell grep avx2 /proc/cpuinfo)
CXXFLAGS += $(if $(AVX_ENABLED),-mavx2)

HEADERS = include/threadpool.h include/avx.h include/tensor/tensor.h \
	include/layers/layer.h include/layers/conv2d.h include/layers/relu.h \
	include/layers/maxpool2d.h include/layers/linear.h include/layers/reshape.h \
	include/layers/bias.h

all: nn-model nn-features nn-tsne

nn-model: data/alexnet.dat $(addprefix data/pca-, $(addsuffix .dat, $(features)))

nn-features: data/features/nn-raw.dat $(addprefix data/features/nn-, $(addsuffix .dat, $(features)))

nn-tsne: data/tsne/nn-raw.png $(addprefix data/tsne/nn-, $(addsuffix .png, $(features)))

data/tsne/nn-raw.png: scripts/tsne.py data/features/nn-raw.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/nn-raw.dat data/filelists.txt $@

data/tsne/nn-%.png: scripts/tsne.py data/features/nn-%.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/nn-$*.dat data/filelists.txt $@

data/features/nn-%.dat: feature data/pca-%.dat data/features/nn-raw.dat
	mkdir -p data/features
	./feature -p data/pca-$*.dat data/features/nn-raw.dat -b -v -o $@

data/pca-%.dat: scripts/gen_pca.py data/features/nn-raw.dat
	mkdir -p data
	python $< $* data/features/nn-raw.dat $@

data/features/nn-raw.dat: feature data/alexnet.dat data/filelists.txt
	mkdir -p data/features
	./feature -a data/alexnet.dat $$(cut -f 1 data/filelists.txt) -b -v -o $@

feature: feature.cpp $(HEADERS)
	$(CXX) $< -o $@ $(CXXFLAGS)

data/alexnet.dat: scripts/gen_alexnet.py
	mkdir -p data
	python $< $@

data/filelists.txt: scripts/gen_filelists.py
	mkdir -p data
	python $< image $@

clean:
	rm feature data -rf

.PHONY: all clean nn-model nn-features nn-tsne
