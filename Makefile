PDF       = Sync3000.pdf
SRCDIR    = src
SRC       = $(SRCDIR)/main.tex
TEXSRCS   = $(wildcard $(SRCDIR)/**/*.tex)

LATEXMK_OPTS = -pdf -bibtex \
               -interaction=nonstopmode \
               -usepretex="\PassOptionsToPackage{draft}{graphicx}" \
               -pdflatex="pdflatex %O %S" \
               -output-directory=build \
	       -jobname=Sync3000

.PHONY: all draft clean

all: $(PDF)

$(PDF): $(TEXSRCS)
	latexmk -f -gg $(LATEXMK_OPTS) $(SRC)

clean:
	latexmk -c $(SRC)
	rm -rf build
