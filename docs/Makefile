# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ZhuSuan
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile api

api:
	@echo "0"
	sphinx-apidoc -fMeT ../zhusuan -o api
	@echo "1"
	for f in api/*.rst; do\
        	perl -pi -e 's/(module|package)$$// if $$. == 1' $$f ;\
	done
	python ./post_apidoc.py
	@echo "Auto-generation of API documentation finished. " \
		"The generated files are in 'api/'"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
