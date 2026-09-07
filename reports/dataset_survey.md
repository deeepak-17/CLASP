# CLASP-P5 · Dataset Survey (D1 Corpus Selection)

_Aditya S (P5 - Eval & Data / Data & Demo Lead)_  
**Generated:** 2026-08-08T17:52:50Z

## 1. Decision

- **Survey:** CLASP D1 corpus selection
- **Selected corpus:** Curated multi-repo OSS Python (D1) (`curated_oss_python`)
- **Weighted score:** 5.0 / 5 (100.0%)
- **Rank:** 1 of 6
- **Licensing:** BSD-3-Clause / Apache-2.0, verified per repository
- **Approx. size:** ~6 repositories, ~2-5k Python files

D1 is the only candidate that scores maximally on the two criteria CLASP cannot compromise on: an unambiguous project boundary to define clusters, and per-file licence certainty consistent with the on-premise premise. The large pretraining corpora are rejected on reproducibility and scale rather than quality; a real proprietary codebase is rejected only on availability, and D1 is explicitly a proxy for it.

## 2. Evaluation criteria

Weights encode CLASP's actual constraints: an unambiguous project boundary is the precondition for cluster-level LoRA, and per-file licence certainty is the precondition for the on-premise enterprise framing.

| Criterion | Weight | Why it matters |
| --- | --- | --- |
| Multi-project structure | 0.25 | CLASP trains one cluster-LoRA per project. The corpus must carry a clean, unambiguous project boundary; a flat bag of functions cannot be partitioned non-IID by project at all. |
| Non-IID suitability | 0.20 | Projects must differ in idiom, API surface and style. If every shard looks alike, FedProx's proximal term has nothing to correct for and the central hypothesis is untestable. |
| License clarity | 0.20 | CLASP's premise is on-premise enterprise use. Every file must carry a permissive, individually verifiable licence; aggregate corpora with mixed or unstated licensing are disqualifying. |
| Offline reproducibility | 0.15 | The corpus must pin to an exact commit, fit on a workstation, and rebuild identically without a data-platform account. |
| Size manageability | 0.10 | Must train a 6.7B model with LoRA on available GPUs within a weekly iteration cycle. |
| Code quality | 0.10 | Maintained, tested, idiomatic Python. Low-quality code produces a weak personalisation signal and noisy Pass@k deltas. |

## 3. Candidate ranking

| Rank | Candidate | Kind | Multi-project structure | Non-IID suitability | License clarity | Offline reproducibility | Size manageability | Code quality | Weighted | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Curated multi-repo OSS Python (D1) | self-assembled | 5 | 5 | 5 | 5 | 5 | 5 | 5.000 | yes |
| 2 | Real proprietary enterprise codebase | ideal but unavailable | 5 | 5 | 1 | 1 | 3 | 4 | 3.300 |  |
| 3 | CodeSearchNet | public benchmark corpus | 2 | 3 | 3 | 4 | 3 | 3 | 2.900 |  |
| 4 | The Stack v2 (BigCode) | large-scale pretraining corpus | 3 | 4 | 3 | 2 | 1 | 3 | 2.850 |  |
| 5 | py150 (ETH SRI) | static-analysis dataset | 3 | 3 | 2 | 3 | 4 | 2 | 2.800 |  |
| 6 | CodeParrot github-code | large-scale pretraining corpus | 2 | 3 | 2 | 2 | 1 | 2 | 2.100 |  |

Raw scores are 1–5 (higher is better).

## 4. Candidate notes

### 1. Curated multi-repo OSS Python (D1)

- **Key:** `curated_oss_python`
- **Kind:** self-assembled
- **Source:** https://github.com/pallets, https://github.com/psf
- **Approx. size:** ~6 repositories, ~2-5k Python files
- **Licensing:** BSD-3-Clause / Apache-2.0, verified per repository
- **Weighted score:** 5.000 (100.0%)

Repository boundaries are exactly the cluster boundaries CLASP needs, with no heuristic inference required. Each repo pins to a commit SHA, licences are per-repo and permissive, and total size fits a laptop.

### 2. Real proprietary enterprise codebase

- **Key:** `proprietary_enterprise`
- **Kind:** ideal but unavailable
- **Source:** n/a
- **Approx. size:** unknown
- **Licensing:** Not obtainable for an academic project
- **Weighted score:** 3.300 (66.0%)

The deployment target CLASP is designed for, and therefore the ideal evaluation substrate. Recorded here to make explicit that D1 is a deliberate proxy for it, not a first choice. No industry partner and no lawful access within Phase II.

### 3. CodeSearchNet

- **Key:** `codesearchnet`
- **Kind:** public benchmark corpus
- **Source:** https://github.com/github/CodeSearchNet
- **Approx. size:** ~2M functions, ~500k documented (6 languages)
- **Licensing:** Mixed OSS licences; per-function provenance retained
- **Weighted score:** 2.900 (58.0%)

Function-level granularity destroys the intra-project context CLASP is built to exploit; reconstructing project boundaries from provenance metadata is possible but lossy and unverifiable.

### 4. The Stack v2 (BigCode)

- **Key:** `the_stack_v2`
- **Kind:** large-scale pretraining corpus
- **Source:** https://huggingface.co/datasets/bigcode/the-stack-v2
- **Approx. size:** >3B files, ~900B tokens
- **Licensing:** Permissive-filtered, but heterogeneous at scale
- **Weighted score:** 2.850 (57.0%)

Overwhelming for a 16-week project on workstation GPUs. Gated access and opt-out churn mean a rebuild months later is not guaranteed to reproduce, which breaks the Phase-III ablation requirement.

### 5. py150 (ETH SRI)

- **Key:** `py150`
- **Kind:** static-analysis dataset
- **Source:** https://www.sri.inf.ethz.ch/py150
- **Approx. size:** 150k Python files
- **Licensing:** Research use; redistribution terms restrictive
- **Weighted score:** 2.800 (56.0%)

Python 2 era code, ASTs rather than raw source in the canonical release, and research-only terms that conflict with CLASP's enterprise-deployment framing.

### 6. CodeParrot github-code

- **Key:** `codeparrot_github_code`
- **Kind:** large-scale pretraining corpus
- **Source:** https://huggingface.co/datasets/codeparrot/github-code
- **Approx. size:** ~115M files, ~1TB
- **Licensing:** Filtered to permissive, but not individually verified
- **Weighted score:** 2.100 (42.0%)

Same scale objection as The Stack, with weaker deduplication and no stable per-project grouping key.

## 5. Known limitations of the selected corpus

Recorded here so they can be cited as threats to validity in the Phase-II report rather than discovered at review time.

- Six OSS libraries are a proxy for an enterprise monorepo estate; generalisation to closed-source style is an assumption, not a result.
- Four of six repositories are Pallets projects, so inter-project distance is smaller than a true multi-vendor estate would exhibit.
- Per-developer partitioning is simulated from directory structure, not from real git authorship.
- The corpus is Python-only; multi-language federation is out of Phase-II scope.

---

Generated by `corpus/survey.py` from `configs/dataset_survey.yaml`. Edit the config and re-run `python scripts/survey_datasets.py` to regenerate; do not hand-edit this file.
