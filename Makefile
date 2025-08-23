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
SIMDIR    = simulation
SIMENV    = $(SIMDIR)/env
SIMPY     = $(SIMENV)/bin/python
SIMPIP    = $(SIMENV)/bin/pip
SIMREQ    = $(SIMDIR)/requirements.txt
SIMSCRIPT = $(SIMDIR)/RFFG.sim.py
SIMSTAMP  = $(SIMDIR)/.sim.ok   # stamp file to indicate the simulation ran successfully

.PHONY: all draft clean sim

all: $(PDF)

# Make the PDF depend on the simulation stamp, so simulation runs first (when needed)
$(PDF): $(SIMSTAMP) $(TEXSRCS)
	latexmk -f -gg $(LATEXMK_OPTS) $(SRC)

# ---- Create Python virtual environment if it does not exist
$(SIMPY):
	python3 -m venv $(SIMENV)

# ---- Install/upgrade requirements (re-run only if requirements.txt changes)
$(SIMENV)/.deps: $(SIMREQ) | $(SIMPY)
	$(SIMPIP) install --upgrade pip
	$(SIMPIP) install -r $(SIMREQ)
	touch $(SIMENV)/.deps

# ---- Run the simulation and write a stamp when successful
# Re-run if requirements or the simulation script changed, or if the stamp is missing
$(SIMSTAMP): $(SIMENV)/.deps $(SIMSCRIPT)
	$(SIMPY) $(SIMSCRIPT)
	touch $(SIMSTAMP)

# Convenience target to run only the simulation
sim: $(SIMSTAMP)
	@echo "Simulation done -> $(SIMSTAMP)"

clean:
	latexmk -c $(SRC)
	rm -rf build
