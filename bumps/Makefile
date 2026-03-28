PAPER = bumps
NEW_PAPER = distributed_bumps
FIGS  = fig_rank1.pdf fig_rank2.pdf fig_rank1_grid.pdf fig_rank2_grid.pdf fig_two_rank1.pdf

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

$(NEW_PAPER).pdf: $(NEW_PAPER).tex
	pdflatex -interaction=nonstopmode $(NEW_PAPER).tex
	pdflatex -interaction=nonstopmode $(NEW_PAPER).tex

fig_%.pdf: fig_%.py
	python3 $<

clean:
	rm -f $(PAPER).{aux,log,out,pdf,bbl,blg} $(NEW_PAPER).{aux,log,out,pdf,bbl,blg} $(FIGS)

.PHONY: all clean
