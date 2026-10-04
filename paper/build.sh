#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bib="${BIBTEX:-bibtex}"
if ! command -v "$bib" >/dev/null; then bib=bibtex.original; fi
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > build_pass1.log
"$bib" paper > bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > build_pass2.log
pdflatex -interaction=nonstopmode -halt-on-error paper.tex > build_pass3.log
if grep -Eq 'Warning:|Overfull|Underfull|^!' paper.log; then
  grep -E 'Warning:|Overfull|Underfull|^!' paper.log
  exit 1
fi
printf 'Clean build: paper.pdf\n'
