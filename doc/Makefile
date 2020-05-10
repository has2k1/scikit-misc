# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = scikit-misc
SOURCEDIR     = .
PAPER         =
BUILDDIR      = _build

# For make theme
theme = cerulean
brand-primary = \#194f7f
success = \#0e6957
text-color = \#333333
headings-color = \#333231
headings-font-weight = bold
font-family-sans-serif = '"Lato", "proxima-nova", "Helvetica Neue", Arial, sans-serif'

# Colors for echo
blue:=$(shell tput setaf 4)
reset:=$(shell tput sgr0)      # reset color codes

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "${blue}  theme${reset} to create a custom sphinx boostrap theme"

.PHONY: help Makefile theme

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

theme:
	../tools/build_theme.sh --theme $(theme) \
	   --brand-primary '$(brand-primary)' \
	   --brand-success '$(success)' \
	   --text-color '$(text-color)' \
	   --headings-color '$(headings-color)' \
	   --headings-font-weight $(headings-font-weight) \
	   --font-family-sans-serif '$(font-family-sans-serif)'
