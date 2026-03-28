PAPER = bumps
FIGS  = fig_rank1.pdf fig_rank2.pdf fig_rank1_grid.pdf fig_rank2_grid.pdf

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

fig_%.pdf: fig_%.py
	python3 $<

clean:
	rm -f $(PAPER).{aux,log,out,pdf,bbl,blg} $(FIGS)

.PHONY: all clean
