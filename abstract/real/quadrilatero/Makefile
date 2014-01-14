DATADIR=/home/leo/dat/boa6/ftg/rawdata

all: estimate.pickle


data.xyz: fetchdata.py ${DATADIR}/BOA6_FTG.XYZ
	python fetchdata.py

estimate.pickle: data.xyz seeds.xyz run.py
	python run.py

plot: estimate.pickle plot.py
	python plot.py

clean:
	rm data.xyz data.pdf estimate.pickle fit.pdf predicted.xyz
