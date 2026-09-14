DECK := presentation
MIKTEX_BIN := $(subst \,/,$(LOCALAPPDATA))/Programs/MiKTeX/miktex/bin/x64
ifeq ($(OS),Windows_NT)
PDFLATEX ?= $(MIKTEX_BIN)/pdflatex.exe
else
PDFLATEX ?= pdflatex
endif
TEXFLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf clean distclean

all: pdf

pdf: $(DECK).pdf

# Two passes: the first writes the frame count the fraction numbering needs.
$(DECK).pdf: $(DECK).tex
	$(PDFLATEX) $(TEXFLAGS) $(DECK).tex
	$(PDFLATEX) $(TEXFLAGS) $(DECK).tex

clean:
	$(RM) $(DECK).aux $(DECK).log $(DECK).nav $(DECK).out $(DECK).snm $(DECK).toc $(DECK).vrb

distclean: clean
	$(RM) $(DECK).pdf
