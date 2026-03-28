PAPER = bumps
FIGS  = fig_rank1.pdf fig_rank2.pdf

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

fig_%.pdf: fig_%.py
	python3 $<

clean:
	rm -f $(PAPER).{aux,log,out,pdf,bbl,blg} $(FIGS)

.PHONY: all clean
