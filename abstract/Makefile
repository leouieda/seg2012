NAME=abstract
OUTPUT=abstract_seg2012_uieda_barbosa.pdf
FIGS=real/quadrilatero/fig/quadrilatero.pdf real/redencao/fig/redencao.pdf synthetic/fig/synthetic.pdf
BIBNAME=seg2012.bib
BIBSOURCE=~/bib/bibtex/seg2012.bib
REFNAME=references

all: ${OUTPUT} ${REFNAME}.pdf


${OUTPUT}: ${NAME}.tex ${BIBNAME} ${FIGS}
	pdflatex ${NAME}.tex
	bibtex ${NAME}
	pdflatex ${NAME}.tex
	pdflatex ${NAME}.tex
	mv ${NAME}.pdf ${OUTPUT}

${REFNAME}.pdf: ${REFNAME}.tex
	pdflatex ${REFNAME}.tex
	pdflatex ${REFNAME}.tex
	pdflatex ${REFNAME}.tex

#${BIBNAME}: ${BIBSOURCE}
	#cp $< $@

clean:
	rm ${OUTPUT} *.aux *.bbl *.log *.blg ${REFNAME}.pdf 
