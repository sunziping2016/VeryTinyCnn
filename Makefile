features = 004 008 012 016 020 040 080 200 400

CXXFLAGS = -Iinclude -std=c++11 -O3 -Wall -Wextra -Wno-unused-parameter -lpthread -lX11
AVX_ENABLED = $(shell grep avx2 /proc/cpuinfo)
CXXFLAGS += $(if $(AVX_ENABLED),-mavx2)

HEADERS = include/threadpool.h include/avx.h include/tensor/tensor.h \
	include/layers/layer.h include/layers/conv2d.h include/layers/relu.h \
	include/layers/maxpool2d.h include/layers/linear.h include/layers/reshape.h \
	include/layers/bias.h

all: nn-tsne-plt hist-tsne-plt data/closest_accuracy.txt data/dist/index.html

visual-deploy: data/dist/index.html
	git checkout gh-pages
	git rm feature.cpp include Makefile README.md scripts image -rf
	mv data/dist/* .
	echo data > .gitignore
	git add .
	git commit -m "build on $$(date)"
	git push origin gh-pages
	git checkout master

data/dist/index.html: scripts/visual.html nn-tsne hist-tsne data/filelists.txt scripts/colors.txt
	mkdir -p data/dist/image
	cp scripts/visual.html data/dist/index.html
	cp data/filelists.txt data/dist/images.txt
	paste data/labels.txt scripts/colors.txt > data/dist/labels.txt
	rm -rf data/dist/plots.txt
	for i in tsne/nn-raw.dat $(addprefix tsne/nn-, $(addsuffix .dat, $(features))) \
			 tsne/hist-raw.dat $(addprefix tsne/hist-, $(addsuffix .dat, $(features))); do \
		echo $$i >> data/dist/plots.txt; \
	done
	cp -r data/tsne data/dist
	for i in $$(cut -f 1 data/filelists.txt); do \
		convert -thumbnail 128x128 $$i data/dist/$$i; \
	done

data/closest_accuracy.txt: scripts/closest_accuracy.py data/filelists.txt nn-features hist-features
	mkdir -p data
	python $< data/filelists.txt $(addprefix data/features/nn-, $(addsuffix .dat, $(features))) data/features/nn-raw.dat \
			  $(addprefix data/features/hist-, $(addsuffix .dat, $(features))) data/features/hist-raw.dat | tee $@

nn-model: data/alexnet.dat $(addprefix data/pca/nn-, $(addsuffix .dat, $(features)))

nn-features: data/features/nn-raw.dat $(addprefix data/features/nn-, $(addsuffix .dat, $(features)))

hist-features: data/features/hist-raw.dat $(addprefix data/features/hist-, $(addsuffix .dat, $(features)))

nn-tsne: data/tsne/nn-raw.dat $(addprefix data/tsne/nn-, $(addsuffix .dat, $(features)))

hist-tsne: data/tsne/hist-raw.dat $(addprefix data/tsne/hist-, $(addsuffix .dat, $(features)))

nn-tsne-plt: data/tsne-plt/nn-raw.png $(addprefix data/tsne-plt/nn-, $(addsuffix .png, $(features)))

hist-tsne-plt: data/tsne-plt/hist-raw.png $(addprefix data/tsne-plt/hist-, $(addsuffix .png, $(features)))


data/tsne-plt/nn-%.png: scripts/plot.py data/tsne/nn-%.dat data/filelists.txt
	mkdir -p data/tsne-plt
	python $< data/tsne/nn-$*.dat $@ data/filelists.txt data/labels.txt

data/tsne-plt/hist-%.png: scripts/plot.py data/tsne/hist-%.dat data/filelists.txt
	mkdir -p data/tsne-plt
	python $< data/tsne/hist-$*.dat $@ data/filelists.txt data/labels.txt

data/tsne/nn-raw.dat: scripts/tsne.py data/features/nn-raw.dat
	mkdir -p data/tsne
	python $< 4096 data/features/nn-raw.dat $@

data/tsne/hist-raw.dat: scripts/tsne.py data/features/hist-raw.dat
	mkdir -p data/tsne
	python $< 4096 data/features/hist-raw.dat $@

data/tsne/nn-%.dat: scripts/tsne.py data/features/nn-%.dat
	mkdir -p data/tsne
	python $< $* data/features/nn-$*.dat $@

data/tsne/hist-%.dat: scripts/tsne.py data/features/hist-%.dat
	mkdir -p data/tsne
	python $< $* data/features/hist-$*.dat $@

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

.PHONY: all clean nn-model nn-features nn-tsne hist-features hist-tsne visual-deploy
