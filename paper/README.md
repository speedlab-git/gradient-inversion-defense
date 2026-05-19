# EMNLP 2026 Long Paper

**Title:** Norm-Bounded Federated Updates Stop Language-Guided Gradient Inversion Attacks

## Build

LaTeX is not installed on the DGX. Compile via Overleaf or any local TeX installation:

```bash
# With pdflatex installed locally:
make

# Or upload to Overleaf and select "pdfLaTeX + Bibtex".
```

Required packages: `acl` (provided), `pgfplots`, `tikz`, `booktabs`, `multirow`, `microtype`, `inconsolata`, `times`. All are in TeXLive 2022+ and on Overleaf by default.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | The paper |
| `custom.bib` | All references |
| `acl.sty` | ACL style file (do not modify) |
| `acl_natbib.bst` | Bibliography style (do not modify) |
| `figures/*.tex` | TikZ/PGFPlots figure source |
| `data/*.csv` | Source data for figures (for reference; figures hardcode values) |
| `outline.md` | Pre-writing outline (not part of submission) |

## Switching review modes

Top of `main.tex`:

```latex
\usepackage[review]{acl}    % anonymous, with line numbers (ARR submission)
\usepackage[final]{acl}     % camera-ready
\usepackage[preprint]{acl}  % non-anonymous arXiv
```

## Page budget

Target: 8 pages of content (long paper). References, limitations, and ethical considerations do not count toward the limit.

## Anonymization

The paper is currently anonymized for ARR. No author names, no affiliations, no self-citations of unpublished work. Code release is pointed at an anonymous URL placeholder; replace with the actual lab repo for camera-ready.
