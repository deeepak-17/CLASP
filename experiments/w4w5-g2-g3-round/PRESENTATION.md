# Presentation pack — Edge lane (W3 / W4 / W5)

Everything you need in one file. **Part A** is what to say. **Part B** is what to run.
Commands marked ⚡ are safe to run live (seconds). Commands marked 🐢 take too long — show
the saved result instead.

---

# PART A — the script

## 0 · One-line reminder of the project (30 sec)

> "CLASP personalises a code-completion model for each developer, without their code ever
> leaving their machine. Each project trains a small add-on to the model locally. Those add-ons
> get combined centrally. Nobody sees anyone else's source code."

Three words to define once, then reuse:

- **Base model** — the general code model everyone starts from. Frozen, never changes.
- **Adapter** — a small trained add-on. About 25 MB against a 1.3 GB model. This is the only
  thing that moves over the network.
- **Perplexity** — how surprised the model is by real code. **Lower is better.** If it drops,
  the model has genuinely learned that project's style.

---

## 1 · What the three weeks were for (1 min)

> "Three weeks, three questions.
>
> **W3 — does the machinery work?** Can we train a real adapter on a real codebase, and can we
> combine adapters together correctly?
>
> **W4 — does the full three-layer stack work?** Six projects, two groups, real combining.
>
> **W5 — is it fast enough to actually use, and does one full cycle run end to end?**"

---

## 2 · W3 — the machinery (2 min)

> "Before this, training ran on eight toy functions I wrote by hand. Now it runs on real
> repositories.
>
> Real files aren't toy-sized. A single pandas file is 7,000 lines. So the first job was
> chunking — glue every file in a project into one long stream and cut it into fixed
> 1024-token blocks."

**Point worth making — the bug that would have been invisible:**

> "The order files get read in decides where every block boundary falls. Windows and Linux list
> files in different orders. Same code, different training data, and nothing would have told
> us. We now sort the file list and record a fingerprint of it, so if the dataset ever shifts,
> we see it."

**Then the merge.** This is the part to be confident about:

> "Serving needs one combined adapter: base plus α times the group adapter plus β times the
> project adapter. α and β are dials we can turn.
>
> We prove the maths is right rather than hoping. Two checks:
> if both dials are zero, we must get the original model back **exactly**.
> If the group dial is zero and the project dial is one, we must get the project adapter
> **exactly**.
>
> Not approximately — exactly. Zero difference, to the last bit."

⚡ **Run demo D1 here** (see Part B). It prints `0.000e+00`.

> "If that number weren't zero, every result after this point would be quietly wrong and we'd
> have no way to tell."

---

## 3 · W4 — six projects, two groups (3 min)

> "Six real projects. Three web — Flask, Requests, Werkzeug. Three scientific — NumPy, Pandas,
> scikit-learn. Each trains its own adapter on its own files only."

**Show the table (Part B, D4). Every row improved:**

| project | before | after | change |
|---|---|---|---|
| Flask | 2.452 | 2.268 | −0.184 |
| Requests | 3.463 | 3.269 | −0.194 |
| Werkzeug | 2.828 | 2.652 | −0.176 |
| NumPy | 2.736 | 2.651 | −0.085 |
| Pandas | 2.910 | 2.717 | −0.193 |
| scikit-learn | 2.584 | 2.370 | −0.214 |

> "Six out of six improved on files they were never trained on. Average improvement 0.174
> perplexity. Whole thing runs in 56 minutes on a 4 GB laptop GPU, using 1.75 GB."

**Now the group adapter — the interesting engineering:**

> "A group adapter is not trained on anything. There is no shared pool of code — that would
> defeat the point. It's *computed* from the three project adapters.
>
> The obvious way is to average the two halves of each adapter separately. That is wrong, and
> we measured how wrong."

⚡ **Run demo D2 here.** Two numbers land:

- SVD method: **0.172** error against the true average
- Naive averaging: **1.399**

> "An error above 1.0 means the naive answer is further from the truth than just submitting
> zero. It isn't slightly worse, it's worse than doing nothing. So we reconstruct each project's
> full update, average those properly, then compress back down. That's the method the design
> document specifies, and now we have the number that justifies it."

---

## 4 · W5 — speed and the full cycle (2 min)

> "Personalisation is worthless if it makes the editor feel slow. The target was under 200
> milliseconds of extra delay before the first character appears, and under 2 seconds to switch
> a developer between adapters."

| target | allowed | measured | |
|---|---|---|---|
| extra delay to first token | 200 ms | **66 ms** | PASS |
| time to switch adapter | 2 s | **0.50 s** | PASS |

> "Both comfortably inside budget, on the weakest hardware we'll ever run on."

**Then the full cycle:**

> "One command now runs the whole loop: take the trained adapters, combine each group, build
> each developer's personalised model, measure it, and decide whether it's good enough to ship
> or should be rolled back. It writes a full record of every number."

---

## 5 · The finding — say this clearly, do not bury it (2 min)

> "Now the result I want to be straight about.
>
> We can turn the group layer's dial from 0 up to 1 and measure what happens. **On all six
> projects, the system chose zero.** It turned the group layer off every single time.
>
> And when we force it on — dial at 0.5 — the model gets *worse* on all six. Only slightly,
> about 0.01, but the sign is consistent every time. So the group layer isn't doing nothing.
> It's mildly hurting.
>
> So does group-level sharing not work? We don't know yet, and I can explain exactly why.
>
> The design says a project adapter should be trained *on top of* its group adapter. We
> couldn't do that, because the group adapter didn't exist until this week — you need the
> project adapters first to build it. Chicken and egg.
>
> So every project trained against the plain base model. Each one independently learned the
> general 'this is Python web code' direction. By the time we add the group layer, it's telling
> them something they already know. It has nothing left to contribute.
>
> That was written down as a minor deviation. It isn't minor — it's the single decision the
> project's main claim rests on. And the experiment that settles it is now unblocked: retrain
> the projects on top of the group adapters we just built, and re-measure. That's the next run."

**If the panel pushes — the honest framing:**

> "Either it works and we have our headline result, or it doesn't and we've found a real
> limitation worth reporting. Both are publishable. What we can't do is claim it works without
> running it."

---

## 6 · What's blocked, and on whom (1 min)

| what | who | why it matters |
|---|---|---|
| Retrain projects on top of group adapters | me | Decides whether the core claim holds |
| Repeat-measure the baseline to get a noise band | Aditya | Right now "improved" means "improved by any amount" — not a real threshold |
| HumanEval scoring won't run on Windows | me / Aditya | Half our ship/rollback rule can't be evaluated, so every decision is provisional |
| Step budget rule needs a team decision | everyone | Flat rule gave the biggest project 33× less training per file than the smallest; fixed at no extra cost, but needs sign-off |
| Web group is Flask/Requests/**Werkzeug** — the plan says Django | Aditya | "Two real-world groups" is a paper claim; the swap isn't recorded anywhere |

---

## 7 · Close (20 sec)

> "The machinery is built and proven correct. Six projects train, two groups aggregate, the
> combined model is fast enough. The one open question is whether group-level sharing helps —
> and we now have everything needed to answer it."

---
---

# PART B — how to run it

**Open a terminal here first:**

```bash
cd "C:/Users/admin/Desktop/git/CLASP/services/edge"
```

All commands use the project's own Python: `./.venv/Scripts/python.exe`

### Before you go in — 30-second check

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Expect `84 passed`. Takes ~3.5 min, so run it while you're getting set up, not on stage.

---

## ⚡ D1 — merge is exactly correct *(use in section 2)*

```bash
./.venv/Scripts/python.exe -m edge.merge --client artifacts/round1/client-requests/adapter --cluster artifacts/clusters/web --alpha 0 --beta 1 --out artifacts/demo_identity
```

**Look for:** `self-check: rel 0.000e+00, abs 0.000e+00` and `rank 16`

**Say:** "Group dial at zero, project dial at one. Zero difference — we got the project adapter
back exactly, not approximately."

Then show it *isn't* trivially always zero:

```bash
./.venv/Scripts/python.exe -m edge.merge --client artifacts/round1/client-requests/adapter --cluster artifacts/clusters/web --alpha 0.5 --beta 1 --out artifacts/demo_mixed
```

**Look for:** `rank 32`, and a tiny non-zero error (~1e-07 — that's just floating-point dust).

---

## ⚡ D2 — SVD vs naive averaging *(use in section 3)*

```bash
./.venv/Scripts/python.exe -m edge.aggregate --clients artifacts/round1/client-flask/adapter artifacts/round1/client-requests/adapter artifacts/round1/client-werkzeug/adapter --weights 80 52 170 --cluster-id web --compare-naive --out artifacts/demo_web
```

Takes ~20 s. **Look for the last two lines:**

```
SVD truncation : mean 0.1723 ...
naive baseline : mean 1.3994   <-- ablation, biased by cross terms
```

**Say:** "0.17 versus 1.40. Above 1.0 means worse than returning nothing at all."

---

## ⚡ D3 — the correctness tests *(optional, if asked "how do you know?")*

```bash
./.venv/Scripts/python.exe -m pytest tests/test_merge.py tests/test_aggregate.py -q
```

~35 s, expect `50 passed`. These are the maths-correctness tests specifically.

---

## ⚡ D4 — the six-project results table *(use in section 3)*

```bash
./.venv/Scripts/python.exe -c "import json,glob; [print(f\"{json.load(open(f))['client_id']:32s} {json.load(open(f))['results']['base_held_out']['perplexity']:7.3f} -> {json.load(open(f))['results']['final_held_out']['perplexity']:7.3f}  ({json.load(open(f))['results']['held_out_ppl_delta']:+.3f})\") for f in sorted(glob.glob('artifacts/round1/*/manifest.json'))]"
```

Prints the before/after table instantly. **Safest option: screenshot this now and put it on a
slide** rather than running it live.

---

## ⚡ D5 — speed results *(use in section 4)*

```bash
./.venv/Scripts/python.exe -c "import json; d=json.load(open('../../experiments/w3-edge-lora-composite/results/ttft_results.json')); print('extra delay to first token:', d['worst_overhead_ms'], 'ms  (budget 200)'); print('adapter switch:', d['adapter_swap']['median_s'], 's  (budget 2)'); print('both pass:', d['nfr']['ttft_pass'] and d['nfr']['swap_pass'])"
```

---

## 🐢 Do NOT run live

| what | why | show instead |
|---|---|---|
| `edge.train_client` | 56 min for all six | the D4 table |
| `edge.round` | ~20 min | `artifacts/round1_out/round_manifest.json` |
| `edge.sweep_lr` | 40–70 min | the tables in `../w3-edge-lora-composite/RESULTS.md` |
| `edge.ttft` | needs a free GPU, several min | D5 above |

---

## Check this before you start

The full six-project cycle was still running when this was written. Check:

```bash
ls artifacts/round1_out/round_manifest.json
```

- **File exists** → the full six-project, two-group cycle finished. Read the numbers from it:
  ```bash
  ./.venv/Scripts/python.exe -c "import json; d=json.load(open('artifacts/round1_out/round_manifest.json')); [print(f\"{k:32s} a*={v['best_alpha']:<4g} {v['full_split']['base_ppl']:.3f} -> {v['full_split']['composite_ppl']:.3f} ({v['personalization_delta_ppl']:+.3f})\") for k,v in sorted(d['clients'].items())]; print('cycle minutes:', d['timings']['total_minutes'])"
  ```
**It finished. Here are the numbers — use these:**

| project | best α | base | combined | change | group layer forced on |
|---|---|---|---|---|---|
| NumPy | **0** | 2.736 | 2.653 | −0.083 | +0.011 worse |
| Pandas | **0** | 2.910 | 2.718 | −0.192 | +0.010 worse |
| scikit-learn | **0** | 2.584 | 2.371 | −0.213 | +0.014 worse |
| Flask | **0** | 2.452 | 2.269 | −0.183 | +0.000 |
| Requests | **0** | 3.463 | 3.260 | −0.203 | +0.002 worse |
| Werkzeug | **0** | 2.828 | 2.653 | −0.175 | +0.012 worse |

Full cycle 27.8 min for six projects and two groups (not counting training).
**All six chose α=0** — that is the section 5 finding, and it is unanimous.

---

## If someone asks…

**"Is perplexity the right measure?"**
> "It's the one the design names as primary. The stronger measure — does it complete code the
> way this project actually writes it — is Aditya's, and it's next."

**"Why only 1.3 billion parameters when the paper says 6.7?"**
> "The 6.7B model needs about 3.9 GB just for weights and this card has 4 GB total. It doesn't
> fit. Every number today is on the 1.3B model and isn't comparable to published 6.7B figures.
> Real runs need a bigger GPU — that's a known escalation."

**"How do you know the projects aren't just seeing each other's code?"**
> "Each project trains only on its own folder. We verified all 733 files by hash — zero overlap
> between training and test files, and zero overlap between projects."

**"Why is NumPy's improvement smaller than the others?"**
> "Budget. NumPy has 1,286 blocks of code but trains on 292, about a fifth of one pass. Flask
> gets nearly a full pass. More training would likely close the gap — that's the step-budget
> decision in the blocked list."

**"Can you show it completing code, live?"**
> "Not today — the honest answer is we measure perplexity, not generated completions yet. A
> side-by-side demo is the natural next deliverable once the eval harness lands."
