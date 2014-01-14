.PHONY = shape, std, plot

all: shape std

shape: estimate-shape.pickle


std: estimate-std.pickle


plot: plot.py
	python plot.py
	
estimate-shape.pickle: run-shape.py data.txt
	python run-shape.py

estimate-std.pickle: run-std.py data.txt
	python run-std.py

data.txt: datagen.py
	python datagen.py

clean:
	rm estimate*.pickle predicted*.xyz seeds.pickle data.txt model.pickle
