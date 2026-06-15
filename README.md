# Project Page — Revisiting Filtered ANN Benchmarks

This branch (`gh-pages`) hosts the **project website** for our paper introducing **α-Hardness** and
**HCBGen**. It contains only the static site; the research code lives on the
[`main`](https://github.com/AIDAS-Lab/FANN_Benchmark/tree/main) branch.

🌐 **Live site:** https://aidas-lab.github.io/FANN_Benchmark/

## Contents

```
index.html        # the page
static/css/        static/js/        static/images/   # assets (figures as SVG)
server.py         # optional local preview server
```

## Preview locally

```bash
# any static server works, e.g.:
python3 -m http.server 8000
# then open http://localhost:8000/index.html
```

> Equations are rendered with [MathJax](https://www.mathjax.org/) (loaded from CDN, so a network
> connection is needed for math to display). The site is adapted from the
> [Nerfies](https://github.com/nerfies/nerfies.github.io) template (CC BY-SA 4.0).

## Deployment

Served via **GitHub Pages** from this branch (`gh-pages`, root). Pushing to `gh-pages` updates the
live site automatically.
