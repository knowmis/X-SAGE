# X-SAGE: eXplainable Situation-Aware Gated Enhancer

A pluggable, interpretable situational re-ranking head for an already-trained recommender. X-SAGE
recognises the situation of each request from its context and the user's recent interactions, and adds
to the backbone score a categorical bias estimated in closed form for that situation, gated by the
certainty of the recognition (Eq. (14) of the paper):

```
ŝ(u, i) = s_B(u, i) + κ · γ_S(v) · Σ_k r_k · b̃^(k)_{c(i)}
```

`s_B` is the backbone score, `r_k` the membership of the request in situation `k`, `b̃^(k)_c` the
standardised bias of category `c` in situation `k` (Eq. (13)), `γ_S(v)` is 1 for a core request and
`1/|T|` for a boundary request between `|T|` situations, and `κ` is selected on validation from
{0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}. With κ = 0 the backbone ranking is returned unchanged.

## Citation

L. Aliberti, G. D'Aniello, M. Gaeta. *X-SAGE for Trustworthy Situation-Aware Recommendation*.
Big Data and Cognitive Computing (MDPI), 2026, accepted; volume and pages to appear.

```bibtex
@article{aliberti2026xsage,
  title   = {{X-SAGE} for Trustworthy Situation-Aware Recommendation},
  author  = {Aliberti, Luca and D'Aniello, Giuseppe and Gaeta, Matteo},
  journal = {Big Data and Cognitive Computing},
  year    = {2026},
  note    = {Accepted; volume and pages to appear}
}
```

## Repository layout

```
xsage/
  l0_sensing.py              window of the user's earlier interactions for each request
  l1_perception.py           context state (per-attribute trees) and intent vector (propagation on W)
  l2_comprehension.py        rough k-means: situations, core and boundary requests
  l3_projection.py           transition matrix between situations (descriptive; not used in scoring)
  descriptor.py              descriptor v = [c̃ ‖ e] for every request
  selection.py               selection of K (silhouette) and ε (boundary fraction)
  recommendation.py          situational bias (Eq. (13)) and combiner (Eq. (14))
  decision.py                decision rule: reliable improvement (*) and confirmed equivalence (†)
  metrics.py                 long-tail item groups
  data.py                    loaders of the processed data and of the backbone scores
  backbones/                 backbone models and score generators
scripts/
  preprocess/                raw data -> data/processed/<dataset>/
  select_perception.py       perception hyper-parameters on validation -> config/params/
  select_kappa.py            κ per backbone and seed on validation -> config/hparams_<dataset>.csv
  run_pipeline.py            head and evaluation on the five datasets -> outputs/results.csv
  ablation_descriptor.py     context-only, intent-only and fused descriptor -> outputs/ablation.csv
config/
  params/<dataset>.json      perception hyper-parameters (γ, n, depth, β, H)
  hparams_<dataset>.csv      selected κ per seed, backbone and method
  foursquare_legacy_taxonomy.json
```

## Datasets

The datasets are not redistributed: download them from the original sources and place the raw files
under `data/raw/`. The data root is the repository root, or the directory in `XSAGE_DATA_ROOT`.

| Dataset | Source | Please cite | Raw files |
|---|---|---|---|
| MovieLens-1M | <https://grouplens.org/datasets/movielens/1m/> | Harper and Konstan, ACM TiiS 5(4), 2015 | `data/raw/ml-1m/` |
| Yelp Open Dataset | <https://www.yelp.com/dataset> | Yelp Open Dataset | `data/raw/yelp_dataset/` |
| Foursquare Global-scale Check-in Dataset (TIST 2015) | <https://sites.google.com/site/yangdingqi/home/foursquare-dataset> | Yang, Zhang and Qu, ACM TIST 7(3), 2016 | `data/raw/foursquare/` |
| KuaiRand-Pure | <https://kuairand.com/> | Gao et al., CIKM 2022 | `data/raw/KuaiRand-Pure/data/` |

Category: first genre (MovieLens-1M), first Yelp root category by priority (Yelp, Philadelphia),
top-level category of the legacy Foursquare taxonomy (Foursquare, New York and São Paulo), first tag of
the video (KuaiRand-Pure). Common to all: iterative 10-core filter, per-user chronological 80/10/10
split, temporal context (hour, day of week, weekend, month) and the category of the user's previous
interaction; Foursquare also uses the geohash of the previous check-in.

## Backbones

| Backbone | Configuration |
|---|---|
| BPR | 64 factors / 150 iterations / lr 0.01 / λ 0.001 |
| EASE | λ = 500 |
| SASRec | d 64, 20 epochs, batch 256, max_len 50, 2 blocks, 1 head |
| FPMC | d 64, 20 epochs, lr 0.05, batch 512 |
| DeepFM and AFM | d 32, 12 epochs, lr 5e-3, L2 1e-5, batch 4096 |
| FM (focal) | d ∈ {32, 64} × {3, 6, 9} epochs, lr 5e-3, selected on validation |

BPR, EASE, SASRec and FPMC are trained with Cornac; DeepFM, AFM and the focal FM are in
`xsage/backbones/` (PyTorch). The focal FM is trained inside `run_pipeline.py` at every seed; the other
six are trained once and their scores are read from `data/<dataset>/backbone/`.

## Installation

```bash
# main environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# backbone environment (Python 3.11): BPR, EASE, FPMC, SASRec with Cornac (numpy 2)
python3.11 -m venv .venv-cornac && source .venv-cornac/bin/activate
pip install -r requirements-backbones.txt
```

## Procedure

From the repository root:

```bash
# 1. preprocess (main environment)
python scripts/preprocess/preprocess_ml1m.py
python scripts/preprocess/preprocess_foursquare.py nyc_tist
python scripts/preprocess/preprocess_foursquare.py saopaulo
python scripts/preprocess/preprocess_yelp.py
python scripts/preprocess/preprocess_kuairand.py

# 2. backbones, for every <ds> in ml1m nyc_tist saopaulo yelp kuairand
python -m xsage.backbones.bpr <ds>              # backbone environment
python -m xsage.backbones.ease <ds>             # backbone environment
python -m xsage.backbones.fpmc <ds>             # backbone environment
python -m xsage.backbones.sasrec <ds>           # backbone environment
python -m xsage.backbones.deepfm_afm <ds> DeepFM
python -m xsage.backbones.deepfm_afm <ds> AFM

# 3. (optional) perception hyper-parameters on validation -> config/params/<ds>.json
python scripts/select_perception.py <ds>

# 4. (optional) κ on validation -> config/hparams_<ds>.csv
python scripts/select_kappa.py <ds>

# 5. head and evaluation, all datasets in one call -> outputs/results.csv
python scripts/run_pipeline.py

# 6. descriptor ablation -> outputs/ablation.csv
python scripts/ablation_descriptor.py
```

The configuration used in the paper is in `config/`, so steps 3 and 4 can be skipped. K and ε are
selected once per dataset with a fixed seed; κ per backbone and seed.

## Statistical protocol

Point estimates are means over five seeds (42–46). Intervals come from a paired bootstrap with users
as clusters (1,500 resamples) on the reference seed. Marks (`xsage/decision.py`, column `mark` of
`outputs/results.csv`): `*` for a favourable sign, an interval excluding zero and the same sign on all
five seeds; `†` for an interval within ±0.005. The Holm correction is reported in `holm_l1` and `holm_l2`.

DeepFM, AFM and SASRec use dropout; their scores, and a few marks, may differ across hardware.

## License

MIT License (see `LICENSE`). The datasets keep their own terms of use.
