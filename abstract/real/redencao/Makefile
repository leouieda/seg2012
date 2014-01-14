DATAFILE=/home/leo/dat/redencao_grav/data/residual_bouguer.xyz

all: estimate.pickle


data.xyz: fetchdata.py ${DATAFILE}
	python fetchdata.py

estimate.pickle: data.xyz seeds.xyz run.py
	python run.py

plot: estimate.pickle plot.py
	python plot.py

clean:
	rm data.xyz outcrop.xyz estimate.pickle seeds.pickle mesh.pickle predicted.xyz
