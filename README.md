# Revisiting Filtered ANN Benchmarks: A Hardness-Controlled Benchmark Generator

Official code for our paper introducing **α-Hardness**, an execution-driven query-level hardness
metric for filtered approximate nearest neighbor (FANN) search, and **HCBGen**, a hardness-controlled
benchmark generator.

> **TL;DR** — A "hybrid workload" is not uniquely determined by its vector dataset: label-synthesis
> choices alone can flip the ranking of methods. We define a principled, execution-grounded hardness
> metric that aligns with real search performance, and use it to generate controllable, realistic, and
> privacy-preserving FANN benchmarks.

📄 [Paper](https://www.vldb.org/pvldb/) · 🌐 [Project Page](https://aidas-lab.github.io/FANN_Benchmark/) · 📊 VLDB 2026

---

## Key Ideas

- **α-Hardness** — models FANN execution as a conditional chain (fetch → scan) and quantifies
  difficulty via the *over-fetch factor* α(q;K). It is index-free, strategy-conditioned, and aligns
  monotonically with empirical recall–QPS (Spearman ρ ≤ −0.7).
- **HCBGen** — uses α-Hardness as an explicit control signal to:
  - **Stress test** with coarse bias modes (High / Low / Random)
  - **Approximate real workloads** by matching a target hardness distribution (Match-PDF)
  - **Share workloads privately** via hardness specifications instead of raw query logs

## Repository Structure

```
.
├── hardness/                       # α-Hardness estimation (standalone)
│   ├── calculate_baseline_hardness.py   # selectivity / correlation proxies
│   ├── calculate_hardness_v5_0.py
│   └── calculate_hardness_v5_1.py       # latest α-Hardness estimator
│
├── HCBGen_v2/                      # Hardness-Controlled Benchmark Generator
│   ├── hardness_aware_generator.py      # main generation entry point
│   ├── hardness_estimator/              # strategy-conditioned hardness scoring
│   ├── label_generator/                 # base / query label synthesis
│   ├── Index_analyzer/                  # per-strategy probing & Pareto comparison
│   │                                    # (ACORN, NHQ, UNG, RWalks, MILVUS, Post/Pre Filter)
│   ├── Benchmark_anal_tool.py
│   └── utils/                           # groundtruth computation, fvecs→bin conversion
│
├── methods/                        # FANN query processing strategies / indices
│   ├── Post_Filtering/  Pre_Filtering/
│   ├── ACORN/  NHQ/  Unified-Navigating-Graph/   # hybrid-native indices
│
├── experiments/                    # scripts to reproduce paper results
│   ├── alignment_experiment/            # hardness ↔ performance alignment (Fig. 3)
│   ├── over_optimism_experiment/        # robustness across workloads (Fig. 6)
│   └── fidelity_experiment/             # Match-PDF proxy fidelity (Fig. 8)
│
└── utils/                          # benchmark / filter / payload generation helpers
    ├── generate_benchmark.py
    ├── make_filter_json.py
    └── make_payload.py
```

## Getting Started

```bash
git clone https://github.com/AIDAS-Lab/FANN_Benchmark.git
cd FANN_Benchmark
```

Dependencies vary per component. The hardness estimation and generator use a standard Python
scientific stack (`numpy`, `scikit-learn`, `faiss`); each index under `methods/` builds from its own
source (see the corresponding subfolder for build instructions).

## Usage

| Goal | Where |
| --- | --- |
| Estimate α-Hardness of a workload | `hardness/calculate_hardness_v5_1.py` |
| Generate a hardness-controlled benchmark | `HCBGen_v2/hardness_aware_generator.py` |
| Probe a strategy's pruning family | `HCBGen_v2/Index_analyzer/` |
| Build / run a FANN index | `methods/<strategy>/` |
| Reproduce a paper figure | `experiments/<name>_experiment/` |

> Each script exposes its own arguments at the top of the file; run with `python <script>.py --help`
> (or inspect the `__main__` block) for the exact options.

## Datasets

| Type | Datasets |
| --- | --- |
| Vector base | SIFT1M, GIST1M, GloVe1M |
| Semi-real | arxiv, LAION1M, TripClick, YFCC |

## Evaluated Strategies

FAISS-based **Post Filtering** and **Pre Filtering**, plus four hybrid-native indices:
**NHQ**, **ACORN**, **UNG**, and **RWalks**.

## Citation

```bibtex
@article{lim2026fann,
  author  = {Lim, Mintaek and Kim, Dogeun and Kim, Minwoo and Do, Jaeyoung},
  title   = {Revisiting Filtered ANN Benchmarks: A Hardness-Controlled Benchmark
             Generator for Realistic Evaluation},
  journal = {Proceedings of the VLDB Endowment (PVLDB)},
  volume  = {14},
  number  = {1},
  year    = {2026}
}
```

## License

Released under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
