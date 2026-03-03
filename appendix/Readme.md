# Appendix Build Workflow

This repository includes a fully reproducible workflow for generating the appendix PDF, including automatically generated parameter tables derived from an Excel file. The process uses:

- a **Makefile**
- a **Python script** (`scripts/build_tab_params.py`)
- a **pixi-managed Python environment**
- **LaTeX** (`appendix.tex`)
- the Excel file **`../data/parameters.xlsx`**

The goal is to ensure that the appendix always reflects the latest parameter values and definitions.

---

## Overview

Building the appendix involves three automated steps:

1. Read parameters from **`parameters.xlsx`**
2. Generate **`tab-params.tex`** using a Python script
3. Compile **`appendix.pdf`** using LaTeX

All Python-related steps run inside the **pixi environment**, ensuring full reproducibility of dependencies.

You normally run everything with:

```bash
make