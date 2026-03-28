PAPER = bumps

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).{aux,log,out,pdf,bbl,blg}

.PHONY: all clean
