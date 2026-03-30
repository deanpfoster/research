PAPER = bumps
NEW_PAPER = distributed_bumps
MART_PAPER = martingale_bumps
FIGS  = fig_rank1.pdf fig_rank2.pdf fig_rank1_grid.pdf fig_rank2_grid.pdf fig_two_rank1.pdf fig_two_rank1_data.pdf
MART_FIGS = fig_martingale_discovery.pdf fig_martingale_snapshot.pdf fig_martingale_1d.pdf fig_martingale_capture.pdf

all: $(PAPER).pdf $(NEW_PAPER).pdf $(MART_PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib $(FIGS)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

$(NEW_PAPER).pdf: $(NEW_PAPER).tex $(NEW_PAPER).bib
	pdflatex -interaction=nonstopmode $(NEW_PAPER).tex
	bibtex $(NEW_PAPER)
	pdflatex -interaction=nonstopmode $(NEW_PAPER).tex
	pdflatex -interaction=nonstopmode $(NEW_PAPER).tex

$(MART_PAPER).pdf: $(MART_PAPER).tex $(NEW_PAPER).bib $(MART_FIGS)
	pdflatex -interaction=nonstopmode $(MART_PAPER).tex
	-bibtex $(MART_PAPER)
	pdflatex -interaction=nonstopmode $(MART_PAPER).tex
	pdflatex -interaction=nonstopmode $(MART_PAPER).tex

fig_martingale_discovery.pdf fig_martingale_snapshot.pdf: fig_martingale.py
	python3 $<

fig_%.pdf: fig_%.py
	python3 $<

clean:
	rm -f $(PAPER).{aux,log,out,pdf,bbl,blg} $(NEW_PAPER).{aux,log,out,pdf,bbl,blg} $(MART_PAPER).{aux,log,out,pdf,bbl,blg} $(FIGS) $(MART_FIGS)

.PHONY: all clean
