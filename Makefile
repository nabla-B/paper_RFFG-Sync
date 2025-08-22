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

# ---------------- Simulation / Python ----------------
SIMDIR   = simulation
SIMENV   = $(SIMDIR)/env
SIMPY    = $(SIMENV)/bin/python
SIMPIP   = $(SIMENV)/bin/pip
SIMREQ   = $(SIMDIR)/requirements.txt
SIMSCRIPT= $(SIMDIR)/RFFG.sim.py

.PHONY: all draft clean sim

all: $(PDF)

$(PDF): $(TEXSRCS)
	latexmk -f -gg $(LATEXMK_OPTS) $(SRC)

# ---- Create Python virtual environment if it does not exist
$(SIMPY):
	python3 -m venv $(SIMENV)

# ---- Install/upgrade requirements (re-run only if requirements.txt changes)
$(SIMENV)/.deps: $(SIMREQ) | $(SIMPY)
	$(SIMPIP) install --upgrade pip
	$(SIMPIP) install -r $(SIMREQ)
	touch $(SIMENV)/.deps

# ---- Run the simulation: ensure venv, install deps, execute script
sim: $(SIMENV)/.deps
	$(SIMPY) $(SIMSCRIPT)

clean:
	latexmk -c $(SRC)
	rm -rf build
