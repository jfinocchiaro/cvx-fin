#!/bin/bash
pdflatex main.tex
pdflatex main.tex
pdfjam --nup 2x4 main.pdf

