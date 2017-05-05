features = 004 008 012 016 020 040 080 200

CXXFLAGS = -Iinclude -std=c++11 -O3 -Wall -Wextra -Wno-unused-parameter -lpthread -lX11
AVX_ENABLED = $(shell grep avx2 /proc/cpuinfo)
CXXFLAGS += $(if $(AVX_ENABLED),-mavx2)

HEADERS = include/threadpool.h include/avx.h include/tensor/tensor.h \
	include/layers/layer.h include/layers/conv2d.h include/layers/relu.h \
	include/layers/maxpool2d.h include/layers/linear.h include/layers/reshape.h \
	include/layers/bias.h

all: nn-tsne hist-tsne

nn-model: data/alexnet.dat $(addprefix data/pca/nn-, $(addsuffix .dat, $(features)))

nn-features: data/features/nn-raw.dat $(addprefix data/features/nn-, $(addsuffix .dat, $(features)))

hist-features: data/features/hist-raw.dat $(addprefix data/features/hist-, $(addsuffix .dat, $(features)))

nn-tsne: data/tsne/nn-raw.png $(addprefix data/tsne/nn-, $(addsuffix .png, $(features)))

hist-tsne: data/tsne/hist-raw.png $(addprefix data/tsne/hist-, $(addsuffix .png, $(features)))

data/tsne/nn-raw.png: scripts/tsne.py data/features/nn-raw.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/nn-raw.dat $@ data/filelists.txt data/labels.txt

data/tsne/hist-raw.png: scripts/tsne.py data/features/hist-raw.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/hist-raw.dat $@ data/filelists.txt data/labels.txt

data/tsne/nn-%.png: scripts/tsne.py data/features/nn-%.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/nn-$*.dat $@ data/filelists.txt data/labels.txt

data/tsne/hist-%.png: scripts/tsne.py data/features/hist-%.dat data/filelists.txt
	mkdir -p data/tsne
	python $< data/features/hist-$*.dat $@ data/filelists.txt data/labels.txt

data/features/nn-%.dat: feature data/pca/nn-%.dat data/features/nn-raw.dat
	mkdir -p data/features
	./feature -p data/pca/nn-$*.dat data/features/nn-raw.dat -b -v -o $@

data/features/hist-%.dat: scripts/pca.py data/features/hist-raw.dat
	mkdir -p data/features
	python $< $* data/features/hist-raw.dat /dev/null $@

data/pca/nn-%.dat: scripts/pca.py data/features/nn-raw.dat
	mkdir -p data/pca
	python $< $* data/features/nn-raw.dat $@

data/features/nn-raw.dat: feature data/alexnet.dat data/filelists.txt
	mkdir -p data/features
	./feature -a data/alexnet.dat $$(cut -f 1 data/filelists.txt) -b -v -o $@

data/features/hist-raw.dat: scripts/feature-hist.py data/filelists.txt
	mkdir -p data/features
	python $< $@ $$(cut -f 1 data/filelists.txt)

feature: feature.cpp $(HEADERS)
	$(CXX) $< -o $@ $(CXXFLAGS)

data/alexnet.dat: scripts/gen_alexnet.py
	mkdir -p data
	python $< $@

data/filelists.txt: scripts/gen_filelists.py
	mkdir -p data
	python $< image $@ data/labels.txt

clean:
	rm feature data -rf

.PHONY: all clean nn-model nn-features nn-tsne hist-features hist-tsne
