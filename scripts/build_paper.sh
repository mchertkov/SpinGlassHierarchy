#!/bin/bash
# Compile paper/main.pdf (needs pdflatex and bibtex).
cd "$(dirname "$0")/../paper"
pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
grep -E "undefined|Overfull" main.log || echo "paper/main.pdf built cleanly"
