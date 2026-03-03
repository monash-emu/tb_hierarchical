# --- Config ---
TEX     := appendix.tex
PDF     := appendix.pdf
TAB     := tab-params.tex
SCRIPT  := scripts/build_tab_params.py
DATA    := ../data/parameters.xlsx

# Default target
all: $(PDF)

# Generate the LaTeX table from Excel using pixi's Python
$(TAB): $(DATA) $(SCRIPT)
	@echo ">>> Generating $(TAB) (using pixi environment)"
	pixi run python $(SCRIPT) --xlsx $(DATA) --out $(TAB) --caption "Model parameters" --label tab-params

# Build the PDF using pixi's environment (assuming latexmk is available system-wide)
$(PDF): $(TEX) $(TAB)
	@echo ">>> Building PDF (via pixi environment)"
	pixi run latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# Clean auxiliary files (still via pixi so latexmk is the same binary)
clean:
	pixi run latexmk -C
	@rm -f $(TAB)

.PHONY: all clean