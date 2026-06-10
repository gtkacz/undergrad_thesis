PAPER_DIR := paper/final
PAPER_MAIN := main
PAPER_PDF := $(PAPER_DIR)/$(PAPER_MAIN).pdf
TEX_IMAGE ?= docker.io/texlive/texlive:latest
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
CONTAINER_USER_ARGS = $(if $(findstring podman,$(notdir $(CONTAINER_ENGINE))),--userns keep-id,--user $(shell id -u):$(shell id -g))
CONTAINER_VOLUME_SUFFIX = $(if $(findstring podman,$(notdir $(CONTAINER_ENGINE))),:Z,)
LATEX_SEQUENCE = pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_MAIN).tex && bibtex $(PAPER_MAIN) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_MAIN).tex && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_MAIN).tex

.PHONY: paper paper/final final paper-container clean-paper help

paper: $(PAPER_PDF)

paper/final: paper

final: paper

$(PAPER_PDF): $(PAPER_DIR)/$(PAPER_MAIN).tex $(PAPER_DIR)/refs.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER_MAIN).tex; \
	elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then \
		cd $(PAPER_DIR) && \
		$(LATEX_SEQUENCE); \
	elif [ -n "$(CONTAINER_ENGINE)" ]; then \
		$(MAKE) paper-container; \
	else \
		printf '%s\n' \
			'No LaTeX compiler found.' \
			'Install latexmk/pdflatex, or install Podman/Docker and rerun make paper.'; \
		exit 127; \
	fi

paper-container:
	@if [ -z "$(CONTAINER_ENGINE)" ]; then \
		printf '%s\n' 'No container engine found. Install Podman or Docker.'; \
		exit 127; \
	fi
	$(CONTAINER_ENGINE) run --rm $(CONTAINER_USER_ARGS) \
		-v "$(CURDIR)/$(PAPER_DIR):/work$(CONTAINER_VOLUME_SUFFIX)" \
		-w /work \
		$(TEX_IMAGE) \
		sh -lc 'if command -v latexmk >/dev/null 2>&1; then latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER_MAIN).tex; else $(LATEX_SEQUENCE); fi'

clean-paper:
	@if command -v latexmk >/dev/null 2>&1; then \
		cd $(PAPER_DIR) && latexmk -C $(PAPER_MAIN).tex; \
	else \
		cd $(PAPER_DIR) && rm -f \
			$(PAPER_MAIN).aux \
			$(PAPER_MAIN).bbl \
			$(PAPER_MAIN).blg \
			$(PAPER_MAIN).fdb_latexmk \
			$(PAPER_MAIN).fls \
			$(PAPER_MAIN).log \
			$(PAPER_MAIN).out \
			$(PAPER_MAIN).pdf \
			$(PAPER_MAIN).spl; \
	fi

help:
	@printf '%s\n' \
		'Targets:' \
		'  make paper        Compile paper/final/main.tex' \
		'  make paper/final  Alias for make paper' \
		'  make final        Alias for make paper' \
		'  make paper-container  Compile via Podman/Docker TeX Live image' \
		'  make clean-paper  Remove paper build artifacts'
