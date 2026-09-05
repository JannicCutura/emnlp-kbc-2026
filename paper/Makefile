PAPER := paper
MIKTEX_BIN := $(subst \,/,$(LOCALAPPDATA))/Programs/MiKTeX/miktex/bin/x64
ifeq ($(OS),Windows_NT)
PDFLATEX ?= $(MIKTEX_BIN)/pdflatex.exe
BIBTEX ?= $(MIKTEX_BIN)/bibtex.exe
else
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
endif
TEXFLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf clean distclean help

all: pdf

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex custom.bib acl.sty acl_natbib.bst
	$(PDFLATEX) $(TEXFLAGS) $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(PDFLATEX) $(TEXFLAGS) $(PAPER).tex
	$(PDFLATEX) $(TEXFLAGS) $(PAPER).tex

clean:
	$(RM) $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log
	$(RM) $(PAPER).out $(PAPER).toc $(PAPER).fdb_latexmk
	$(RM) $(PAPER).fls $(PAPER).synctex.gz

distclean: clean
	$(RM) $(PAPER).pdf

help:
	@echo "make          Build paper.pdf with pdfLaTeX and BibTeX"
	@echo "make clean    Remove intermediate LaTeX files"
	@echo "make distclean Remove intermediate files and paper.pdf"
