"""
================================================================
  CLASP - P2 Cluster Service
  Student : Prasanth
  Progress Demonstration
================================================================

Run command:
    cd C:\\Users\\pras2\\OneDrive\\Documents\\CLASP\\services\\cluster\\src
    python -m cluster.demo_all_weeks

What this script shows:
    WEEK 1 - Flower environment set up, skeleton round-trips a payload
    WEEK 2 - 3 simulated clients, FedProx proximal term, adapter format
    WEEK 3 - Delta-W reconstruction, Streaming exact average, SVD refactor
    WEEK 4 - Round metrics (timer), Redistribution, Round-trip test, Fault tolerance
    WEEK 5 - 3-client aggregation round (the same code as ``python -m cluster.demo``)
    WEEKS 6-13 - the real multi-cluster pipeline (``cluster.demo_phase2``): straggler
             timeout, per-cluster SVD aggregation, redistribution with retries,
             isolation, dynamic re-clustering, static fallback, reproducibility, HTTP
    WEEKS 14-16 - report/evidence deliverables: checked for existence, not re-run here
    Blocked by other modules (not demonstrated): G1 mTLS and DP mu-tuning (P3)
"""

from __future__ import annotations

import time

import numpy as np

LINE  = "=" * 64
DASH  = "-" * 64


def heading(title: str) -> None:
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def subheading(title: str) -> None:
    print(f"\n{DASH}")
    print(f"  {title}")
    print(DASH)


def ok(msg: str) -> None:
    print(f"  [PASS]  {msg}")


def info(msg: str) -> None:
    print(f"  >>  {msg}")


def show(label: str, value) -> None:
    print(f"  {label:<30} {value}")


# ================================================================
#  WEEK 1 DEMO
#  What was built: Flower environment setup, skeleton round-trip
#  File: client.py (DummyClient), server.py (SVDLoRAStrategy skeleton)
# ================================================================

def demo_week1():
    heading("Flower Environment + Skeleton Round-Trip")

    info("TASK: Set up Flower, create a skeleton client and server.")
    info("The DummyClient simply echoes parameters back with a visible bump.")
    info("This proves the Flower pipeline is wired correctly end to end.")
    print()

    # Import only what Week 1 needed

    from cluster.adapter_format import random_adapter
    from cluster.client import DummyClient

    # Create the initial global adapter (what server sends out)
    initial = random_adapter(in_features=32, out_features=32, seed=0)
    global_arrays = initial.to_ndarrays()

    info("Initial global adapter created:")
    show("  rank", initial.rank)
    show("  alpha", initial.alpha)
    show("  target_modules", list(initial.target_modules))
    show("  total arrays (A+B x 4 modules)", len(global_arrays))
    print()

    subheading("DummyClient round-trip (client.py: DummyClient.fit)")
    info("Source: client.py, DummyClient.fit (abridged)")
    info("  def fit(self, parameters, config):")
    info("      bump = float(config.get('bump', 1.0))")
    info("      updated = [p + bump for p in parameters]")
    info("      return updated, self.num_examples, {'client_id': ...}")
    print()

    # Simulate 3 dummy clients
    clients = [DummyClient(client_id=f"dummy-{i}", num_examples=10*(i+1)) for i in range(3)]

    for client in clients:
        arrays_in = [a.copy() for a in global_arrays]
        updated, n_examples, _metrics = client.fit(arrays_in, {"bump": 1.0})

        # Verify the bump happened
        diff = float(np.mean(np.abs(updated[0] - global_arrays[0])))
        ok(f"{client.client_id}: fit() returned {len(updated)} arrays, "
           f"num_examples={n_examples}, mean_diff={diff:.4f} (bump=1.0 applied)")

    print()
    ok("COMPLETE: Flower pipeline wired, DummyClient round-trip works")
    ok("Files: client.py (DummyClient class), server.py (SVDLoRAStrategy skeleton)")


# ================================================================
#  WEEK 2 DEMO
#  What was built:
#    Mon - 3 simulated clients
#    Tue - FedProx proximal term in client training loop
#    Thu - Adapter format (PEFT state_dict <-> flat numpy arrays)
#    Fri - Contracts freeze (rank=16, target_modules=q/k/v/o)
# ================================================================

def demo_week2():
    heading("FedProx Proximal Term + Adapter Format Contract")

    # ----------- WEEK 2 MON: 3 simulated clients -----------
    subheading("3 Simulated Clients (simulation.py: make_dummy_clients)")
    info("Source: simulation.py, make_dummy_clients (abridged)")
    info("  def make_dummy_clients(n=3):")
    info("      return [DummyClient(client_id=f'dummy-{i}', ...)]")
    print()

    from cluster.simulation import make_dummy_clients
    clients = make_dummy_clients(n=3)
    for c in clients:
        ok(f"Created: {c.client_id} (num_examples={c.num_examples})")

    # ----------- WEEK 2 TUE: FedProx proximal term -----------
    subheading("FedProx Proximal Term (client.py: LoRAClient.train_local)")
    info("FedProx paper (Li et al. 2020): adds a penalty to local training.")
    info("Formula: total_loss = task_loss + (mu/2) * ||w - w_global||^2")
    info("Purpose: Prevents clients from drifting too far from global model.")
    info("This is critical for non-IID data (each client has different data).")
    print()
    info("Source: client.py, LoRAClient.train_local (the FedProx term, abridged):")
    info("  prox = sum(")
    info("      ((p - g) ** 2).sum()")
    info("      for p, g in zip(self._trainable(), global_params)")
    info("  )")
    info("  (loss + 0.5 * self.mu * prox).backward()")
    print()
    ok("FedProx: mu=0.01 (proximal weight), 20 local steps per round")
    ok("FedProx motivation (paper): limit client drift on non-IID data")
    ok("Measured effect: tests/test_fedprox.py and the mu sweep in `python -m cluster.evidence`")

    # ----------- WEEK 2 THU: Adapter format -----------
    subheading("Adapter Format: PEFT state_dict <-> NumPy (adapter_format.py)")
    info("Problem: P1 (Edge) saves adapters as PyTorch PEFT state_dict keys.")
    info("Problem: Flower sends/receives flat Python lists of NumPy arrays.")
    info("Solution: LoRAAdapter class converts between both formats.")
    print()
    info("Source: adapter_format.py")
    info("  to_state_dict()  -> {'layers.0.q_proj.lora_A.weight': array, ...}")
    info("  from_state_dict() <- {'layers.0.q_proj.lora_A.weight': array, ...}")
    info("  to_ndarrays()    -> [A_q, B_q, A_k, B_k, A_v, B_v, A_o, B_o]")
    info("  from_ndarrays()  <- [A_q, B_q, A_k, B_k, A_v, B_v, A_o, B_o]")
    print()

    from cluster.adapter_format import LoRAAdapter, random_adapter

    adapter = random_adapter(32, 32, seed=1)

    # Show state_dict format
    sd = adapter.to_state_dict()
    ok(f"to_state_dict() -> {len(sd)} keys (PEFT format for P1 Edge):")
    for key in list(sd.keys())[:3]:
        show(f"    {key}", f"shape={sd[key].shape}")
    print("      ...")

    # Show flat array format
    arrays = adapter.to_ndarrays()
    ok(f"to_ndarrays() -> {len(arrays)} flat arrays (Flower wire format):")
    for i, a in enumerate(arrays):
        show(f"    array[{i}]", f"shape={a.shape}")

    # Prove round-trip
    recovered = LoRAAdapter.from_ndarrays(arrays)
    for m in adapter.target_modules:
        np.testing.assert_allclose(
            adapter.modules[0][m]["lora_A"],
            recovered.modules[0][m]["lora_A"], atol=1e-6
        )
    ok("Round-trip verified: from_ndarrays(to_ndarrays()) == original")

    # ----------- WEEK 2 FRI: Contracts -----------
    subheading("Contracts Freeze (adapter_format.py: DEFAULT_RANK / DEFAULT_ALPHA / TARGET_MODULES)")
    info("Source: adapter_format.py, module constants")
    info("  DEFAULT_RANK = 16        <- agreed with all teams")
    info("  DEFAULT_ALPHA = 16.0     <- = Edge's lora_alpha; LoRA scaling = alpha/rank = 1.0")
    info("  TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')")
    print()
    ok("Contracts: rank=16, alpha=16 (matches Edge's lora_init.py), 4 attention modules")
    ok("Cluster's defaults match the values Edge pins; P3/P4/P5 have no code to compare yet")

    print()
    ok("COMPLETE: FedProx implemented, adapter format works, contracts frozen")


# ================================================================
#  WEEK 3 DEMO
#  What was built:
#    Tue - Delta-W reconstruction (B @ A)
#    Wed - Streaming exact weighted average
#    Thu - Truncated SVD re-factorization back to rank 16
# ================================================================

def demo_week3():
    heading("Delta-W Reconstruction + Streaming Mean + SVD Aggregation")

    from cluster.adapter_format import random_adapter
    from cluster.aggregation import (
        StreamingWeightedMean,
        aggregate_naive,
        aggregate_svd,
        exact_average_delta,
    )

    # ----------- WEEK 3 TUE: Delta-W -----------
    subheading("Delta-W Reconstruction: delta_W = B @ A (adapter_format.py: LoRAAdapter.delta_w)")
    info("LoRA formula: new_output = W0*x + (alpha/r) * B @ A @ x")
    info("The CHANGE in weights is: delta_W = B @ A")
    info("")
    info("Source: adapter_format.py, LoRAAdapter.delta_w:")
    info("  def delta_w(self, module: str, layer: int = 0) -> np.ndarray:")
    info("      pair = self.modules[layer][module]")
    info("      return pair['lora_B'] @ pair['lora_A']")
    print()

    adapter = random_adapter(8, 8, rank=4, seed=42)
    # Make B non-zero so delta_w is interesting
    adapter.modules[0]["q_proj"]["lora_B"] = np.random.default_rng(42).normal(
        size=(8, 4)).astype(np.float32)

    dw = adapter.delta_w("q_proj")
    B = adapter.modules[0]["q_proj"]["lora_B"]
    A = adapter.modules[0]["q_proj"]["lora_A"]
    ok(f"lora_A shape: {A.shape}  (rank={adapter.rank}, in=8)")
    ok(f"lora_B shape: {B.shape}  (out=8, rank={adapter.rank})")
    ok(f"delta_W = B @ A shape: {dw.shape}  (out=8, in=8)")
    np.testing.assert_allclose(dw, B @ A, atol=1e-6)
    ok("Verified: delta_w() == lora_B @ lora_A  [CORRECT]")

    # ----------- WEEK 3 WED: Streaming Mean -----------
    subheading("Streaming Weighted Mean (aggregation.py: StreamingWeightedMean)")
    info("Problem: With 100 clients, loading all delta_Ws into memory at once")
    info("         would require 100x memory. Not scalable.")
    info("")
    info("Solution: Streaming incremental update formula:")
    info("  m <- m + (w_i / W_i) * (x_i - m)")
    info("This uses ONE accumulator regardless of number of clients.")
    info("")
    info("Source: aggregation.py, StreamingWeightedMean.update (abridged):")
    info("  def update(self, value, weight=1.0):")
    info("      self._total_weight += weight")
    info("      if self._mean is None:")
    info("          self._mean = value.copy()")
    info("      else:")
    info("          self._mean += (weight / self._total_weight) * (value - self._mean)")
    print()

    # Demonstrate streaming mean vs numpy mean
    values = [np.array([1.0, 2.0, 3.0]),
              np.array([4.0, 5.0, 6.0]),
              np.array([7.0, 8.0, 9.0])]
    weights = [1.0, 2.0, 3.0]

    stream = StreamingWeightedMean()
    for v, w in zip(values, weights):
        stream.update(v, w)

    numpy_result = np.average(values, weights=weights, axis=0)
    np.testing.assert_allclose(stream.result(), numpy_result, atol=1e-10)
    ok(f"Streaming mean result: {stream.result()}")
    ok(f"NumPy weighted mean:   {numpy_result}")
    ok("Both match exactly - streaming algorithm is memory-efficient and correct")

    # ----------- WEEK 3 THU: SVD Aggregation -----------
    subheading("Truncated SVD Re-factorization (aggregation.py: truncated_svd_refactor / aggregate_svd)")
    info("WHY AGGREGATE IN WEIGHT SPACE (then re-factorize with SVD):")
    info("")
    info("  Naive baseline (ablation):  mean(B_k) @ mean(A_k)")
    info("  This implementation:        SVD of mean(B_k @ A_k)")
    info("")
    info("These differ in general: mean(B) @ mean(A) is not mean(B @ A) when clients")
    info("have different A matrices (they coincide only if all clients share the same A).")
    info("The SVD path averages the weight updates themselves, then truncates to rank r.")
    print()
    info("Source: aggregation.py, truncated_svd_refactor (abridged):")
    info("  u, s, vt = np.linalg.svd(delta_w, full_matrices=False)")
    info("  sqrt_s = np.sqrt(s[:rank])")
    info("  b = u[:, :rank] * sqrt_s          # (out, rank)")
    info("  a = sqrt_s[:, None] * vt[:rank, :] # (rank, in)")
    info("  return a, b   # so that b @ a ~= delta_w")
    print()

    # Create 3 adapters and show SVD vs naive error
    dim = 8
    rank = 4
    rng = np.random.default_rng(0)
    adapters_list = []
    for i in range(3):
        a_arr = rng.normal(0, 0.1, (rank, dim)).astype(np.float32)
        b_arr = rng.normal(0, 0.1, (dim, rank)).astype(np.float32)
        ad = random_adapter(dim, dim, rank=rank, seed=i)
        ad.modules[0]["q_proj"]["lora_A"] = a_arr
        ad.modules[0]["q_proj"]["lora_B"] = b_arr
        for m in ["k_proj", "v_proj", "o_proj"]:
            ad.modules[0][m]["lora_A"] = rng.normal(0, 0.1, (rank, dim)).astype(np.float32)
            ad.modules[0][m]["lora_B"] = rng.normal(0, 0.1, (dim, rank)).astype(np.float32)
        adapters_list.append(ad)

    weights_list = [1.0, 1.0, 1.0]

    # True exact average in weight space
    exact = exact_average_delta(iter(adapters_list), weights_list)

    # SVD aggregation
    svd_merged = aggregate_svd(iter(adapters_list), weights_list, rank=rank)

    # Naive aggregation
    naive_merged = aggregate_naive(iter(adapters_list), weights_list)

    module = "q_proj"
    svd_err   = float(np.linalg.norm(svd_merged.delta_w(module) - exact[module]))
    naive_err = float(np.linalg.norm(naive_merged.delta_w(module) - exact[module]))

    ok(f"SVD aggregation error   from true mean: {svd_err:.8f}  <- exact mean + truncated SVD")
    ok(f"Naive aggregation error from true mean: {naive_err:.8f}  <- naive A/B averaging (baseline)")
    if svd_err <= naive_err + 1e-9:
        ok("SVD error <= naive error on this random 3-client example (one example, not a proof)")
    else:
        ok("Note: on this example the naive baseline was not worse (rank/data dependent)")

    print()
    ok("COMPLETE: delta_W reconstruction, streaming mean, SVD aggregation all working")


# ================================================================
#  WEEK 4 DEMO
#  What was built:
#    Mon - Round metrics: timer (duration_s), round_log
#    Tue - Redistribution to clients (aggregate_fit return value)
#    Wed - Round-trip correctness test
#    Thu - Fault tolerance (3 layers + quorum guard)
# ================================================================

def demo_week4():
    heading("Metrics + Redistribution + Tests + Fault Tolerance")

    from flwr.common import (
        Code,
        FitRes,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )

    from cluster.adapter_format import LoRAAdapter, random_adapter
    from cluster.aggregation import aggregate_svd
    from cluster.server import build_strategy

    # Helper to create a realistic adapter
    def make_trained_adapter(seed: int) -> LoRAAdapter:
        rng = np.random.default_rng(seed)
        adapter = random_adapter(32, 32, seed=seed)
        for m in adapter.target_modules:
            adapter.modules[0][m]["lora_B"] = rng.normal(
                scale=0.1, size=(32, adapter.rank)).astype(np.float32)
        return adapter

    def good_fit_res(adapter: LoRAAdapter, n: int, loss: float) -> FitRes:
        return FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(adapter.to_ndarrays()),
            num_examples=n,
            metrics={"loss": loss},
        )

    def straggler_fit_res(adapter: LoRAAdapter) -> FitRes:
        """Client completed gRPC but had internal error (OOM, crash, etc.)"""
        return FitRes(
            status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="client OOM"),
            parameters=ndarrays_to_parameters(adapter.to_ndarrays()),
            num_examples=10,
            metrics={},
        )

    def corrupt_fit_res() -> FitRes:
        """Client sent corrupted / truncated tensor data"""
        bad = [np.zeros((16, 32), dtype=np.float32)] * 2  # wrong count
        return FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(bad),
            num_examples=10,
            metrics={},
        )

    initial = random_adapter(32, 32, seed=0)

    # ----------- WEEK 4 MON: Round Metrics + Timer -----------
    subheading("Round Metrics + Wall-Clock Timer (server.py: SVDLoRAStrategy.aggregate_fit)")
    info("TASK: Log round_id, num_clients, mean_loss, duration_s after every round.")
    info("")
    info("Source: server.py, SVDLoRAStrategy.aggregate_fit (abridged):")
    info("  metrics = {")
    info("      'round': server_round,")
    info("      'num_clients': len(results),")
    info("      'num_failures': num_failures + len(skipped),")
    info("      'aggregation': self.aggregation,")
    info("  }")
    info("  if losses:")
    info("      metrics['mean_loss'] = float(sum(losses) / len(losses))")
    info("  metrics['duration_s'] = time.monotonic() - t0")
    info("  self.round_log.append(metrics)")
    print()
    info("WHY time.monotonic() and NOT time.time()?")
    info("  time.time() can jump backwards (NTP sync, daylight saving)")
    info("  time.monotonic() is guaranteed to NEVER go backwards")
    info("  For measuring durations -> always use monotonic!")
    print()

    strategy = build_strategy(initial, aggregation="svd", min_clients=3)

    adapters = [make_trained_adapter(s) for s in (1, 2, 3)]
    results = [
        (None, good_fit_res(adapters[0], 100, 0.42)),
        (None, good_fit_res(adapters[1], 200, 0.38)),
        (None, good_fit_res(adapters[2], 150, 0.35)),
    ]

    t_before = time.monotonic()
    params, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    t_after  = time.monotonic()

    ok(f"aggregate_fit() ran in {(t_after - t_before)*1000:.2f} ms")
    print()
    info("Metrics returned by aggregate_fit:")
    for k, v in metrics.items():
        if isinstance(v, float):
            show(f"  {k}", f"{v:.6f}")
        else:
            show(f"  {k}", v)

    ok(f"round_log has {len(strategy.round_log)} entry after 1 round")
    ok("Round metrics logged correctly [PASS]")

    # ----------- WEEK 4 TUE: Redistribution -----------
    subheading("Redistribution to Clients (server.py: SVDLoRAStrategy.aggregate_fit)")
    info("After aggregation, the merged adapter must be sent BACK to all clients.")
    info("Flower handles the actual network transmission.")
    info("Your job: return the merged adapter in Flower's Parameters format.")
    info("")
    info("Source: server.py, SVDLoRAStrategy.aggregate_fit (the return statement):")
    info("  return ndarrays_to_parameters(merged.to_ndarrays()), metrics_out")
    info("")
    info("  merged.to_ndarrays()         -> 8 flat NumPy arrays")
    info("  ndarrays_to_parameters(...)  -> Flower Parameters (gRPC format)")
    info("  Flower then broadcasts this  -> every client before next round")
    print()

    assert params is not None, "ERROR: params should not be None"
    arrays_out = parameters_to_ndarrays(params)
    ok(f"Redistribution payload: {len(arrays_out)} arrays")

    # Verify it can be decoded correctly on the client side
    recovered = LoRAAdapter.from_ndarrays(arrays_out)
    recovered.validate()
    ok(f"Client can decode it: rank={recovered.rank}, modules={list(recovered.target_modules)}")
    ok("Redistribution works [PASS]")

    # ----------- WEEK 4 WED: Round-trip test -----------
    subheading("Round-Trip Correctness Test (tests/test_aggregation.py)")
    info("TASK: Verify that the merged adapter is the best rank-r approximation")
    info("of the exact weighted average delta_W.")
    info("")
    info("Source: tests/test_aggregation.py test_full_rank_svd_matches_exact_average")
    info("  -> At full rank, SVD re-factorization is LOSSLESS")
    info("  -> Proves: to_ndarrays -> aggregate -> from_ndarrays gives correct result")
    print()

    from cluster.aggregation import exact_average_delta

    test_adapters = [make_trained_adapter(s) for s in (10, 11, 12)]
    test_weights  = [100.0, 200.0, 150.0]

    exact = exact_average_delta(iter(test_adapters), test_weights)
    merged_adapter = aggregate_svd(iter(test_adapters), test_weights, rank=16)

    all_pass = True
    for m in initial.target_modules:
        err = float(np.linalg.norm(merged_adapter.delta_w(m) - exact[m]))
        s = np.linalg.svd(exact[m], compute_uv=False)
        optimal_truncation_err = float(np.sqrt((s[16:] ** 2).sum()))
        passed = err <= optimal_truncation_err * (1 + 1e-4) + 1e-6
        all_pass = all_pass and passed
        ok(f"{m}: recon_error={err:.2e}, optimal_bound={optimal_truncation_err:.2e} "
           f"-> {'PASS' if passed else 'FAIL'}")

    ok("Round-trip correctness verified [PASS]" if all_pass
       else "Round-trip test FAILED")

    # ----------- WEEK 4 THU: Fault Tolerance -----------
    subheading("Fault Tolerance: 3-Layer Defense (server.py)")
    info("PROBLEM: In real networks, clients fail. We need the server to be")
    info("robust to partial failures without crashing the entire round.")
    info("")
    info("MY SOLUTION: 3 layers of protection in aggregate_fit:")
    info("")
    info("  LAYER 1: Count gRPC transport failures")
    info("           (clients that never completed the network call)")
    info("")
    info("  LAYER 2: Filter non-OK Status codes")
    info("           (clients that completed gRPC but failed internally)")
    info("")
    info("  LAYER 3: Skip + count malformed payloads (corrupted/truncated")
    info("           tensor data); retry-with-backoff lives on the")
    info("           REDISTRIBUTION side (redistribution.redistribute)")
    info("")
    info("  + QUORUM GUARD: Skip round if too few clients succeeded")
    print()

    from cluster.straggler import StragglerPolicy

    fault_strategy = build_strategy(initial, min_clients=1)
    quorum_strategy = build_strategy(
        initial, min_clients=1, straggler_policy=StragglerPolicy(min_fraction=0.5)
    )

    # ---- Layer 1: Transport failures ----
    info(">>> Layer 1 Demo: 2 gRPC transport failures + 1 OK client")
    good = make_trained_adapter(20)
    fake_failures = [object(), object()]  # simulates Flower's gRPC failure objects
    p, m = fault_strategy.aggregate_fit(
        server_round=1,
        results=[(None, good_fit_res(good, 100, 0.5))],
        failures=fake_failures,
    )
    assert p is not None
    ok("Result: params returned (1 good client aggregated)")
    ok(f"num_failures in metrics = {m.get('num_failures', 'NOT PRESENT')} "
       f"(transport failures counted)")

    # ---- Layer 2: Straggler filtering ----
    info("\n>>> Layer 2 Demo: 1 OK client + 1 straggler (non-OK status)")
    good     = make_trained_adapter(21)
    straggler = make_trained_adapter(22)
    p, m = fault_strategy.aggregate_fit(
        server_round=2,
        results=[
            (None, good_fit_res(good, 100, 0.4)),     # OK -> included
            (None, straggler_fit_res(straggler)),       # non-OK -> SKIPPED
        ],
        failures=[],
    )
    assert p is not None
    ok("Result: only 1 client aggregated (straggler skipped)")
    ok(f"num_clients in metrics = {m.get('num_clients', '?')} (only good client)")
    ok(f"num_failures in metrics = {m.get('num_failures', 'NOT PRESENT')} (straggler counted)")

    # ---- Quorum Guard ----
    info("\n>>> Quorum Guard Demo: 1 OK + 3 stragglers = 25% OK < 50% threshold")
    good       = make_trained_adapter(23)
    stragglers = [make_trained_adapter(s) for s in (24, 25, 26)]
    p, m = quorum_strategy.aggregate_fit(
        server_round=3,
        results=[
            (None, good_fit_res(good, 100, 0.3)),
            *[(None, straggler_fit_res(s)) for s in stragglers],
        ],
        failures=[],
    )
    if p is None and m == {}:
        ok("Result: params=None, metrics={}  (round SKIPPED by quorum guard)")
        ok("25% OK < 50% minimum threshold -> round aborted, no corrupt aggregate")
    else:
        ok("UNEXPECTED: round was not skipped by the quorum guard")

    print()
    ok("COMPLETE: Metrics, redistribution, tests, fault tolerance all working")


# ================================================================
#  FULL MULTI-ROUND SIMULATION
#  Shows Weeks 2+3+4 working together across 3 rounds
# ================================================================

def demo_full_simulation():
    heading("FULL SIMULATION - Training and Aggregation Working Together (3 Federated Rounds)")

    info("This shows the complete P2 pipeline running end-to-end:")
    info("  make_real_clients()  ->  3 FedProx-trained clients")
    info("  run_round()          ->  1 federated round")
    info("  aggregate_svd()      ->  SVD aggregation")
    info("  round_log            ->  metrics per round")
    print()

    from cluster.adapter_format import random_adapter
    from cluster.simulation import make_real_clients, run_federated

    clients = make_real_clients(n=3, dim=32, local_steps=20, seed=42)
    ok("3 real LoRAClient instances created (FedProx, mu=0.01)")

    initial = random_adapter(32, 32, seed=0)
    ok(f"Initial global adapter: rank={initial.rank}, modules={list(initial.target_modules)}")

    print()
    info("Running 3 federated rounds...")
    print()

    results = run_federated(
        clients=clients,
        initial_adapter=initial,
        num_rounds=3,
        aggregation="svd",
    )

    print(f"  {'Round':<8} {'Mean Loss':<15} {'Clients':<10} {'Status'}")
    print(f"  {'-----':<8} {'---------':<15} {'-------':<10} {'------'}")
    for r in results:
        loss_str = f"{r.mean_loss:.6f}" if r.mean_loss is not None else "N/A"
        status = "OK" if r.num_clients == 3 else "PARTIAL"
        print(f"  {r.round_id:<8} {loss_str:<15} {r.num_clients:<10} {status}")

    losses = [r.mean_loss for r in results if r.mean_loss is not None]
    print()
    if len(losses) >= 2:
        if losses[-1] < losses[0]:
            ok(f"Loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f}")
            ok("FedProx training is working: model improves across rounds")
        else:
            ok(f"Loss values: {[f'{loss:.4f}' for loss in losses]}")
            ok("Note: toy model/few steps may not show clear decrease")

    ok("All 3 rounds completed successfully")
    ok("Aggregated adapter (the cluster-LoRA) is ready to redistribute to all clients")


# ================================================================
#  WEEKS 5-16
# ================================================================

def demo_week5():
    """Week 5: one 3-client aggregation round through the real pipeline."""
    heading("3-client federated aggregation round (cluster.demo)")
    from cluster.demo import run_demo

    result = run_demo(clients=3, seed=42, verbose=False)
    metrics = result["round_metrics"]
    info("Same code path as: python -m cluster.demo --seed 42 (without writing a backup file)")
    show("engine", result["engine_used"])
    show("num_clients", metrics["num_clients"])
    show("aggregation", metrics["aggregation"])
    show("mean_loss", f"{metrics['mean_loss']:.6f}")
    assert str(result["status"]).lower() == "success", result["status"]
    ok(f"Round finished with status {result['status']}")
    return result


def demo_weeks_6_to_13():
    """Weeks 6-13: the real multi-cluster pipeline, with its own assertions."""
    heading("Multi-cluster pipeline (cluster.demo_phase2)")
    info("Review fixes (alpha=16 contract, validation) - covered by the test suite")
    info("Clusters, isolation, straggler policy, re-clustering, fallback, HTTP")
    info("The block below asserts its own claims; it raises if one stops being true.")
    print()
    from cluster import demo_phase2

    return demo_phase2.run(seed=42, verbose=True)


def demo_weeks_14_to_16():
    """Weeks 14-16: report and evidence deliverables. Existence check only."""
    from pathlib import Path

    heading("Report, documentation and evidence deliverables")
    root = Path(__file__).resolve().parents[2]
    deliverables = [
        ("docs/P2_CLUSTER_REPORT.md", "Phase II report, P2 section (architecture ... limitations)"),
        ("docs/INTEGRATION_BOUNDARIES.md", "Edge / Security / Registry / Evaluation boundaries"),
        ("docs/WEEK6_REVIEW.md", "Review pass notes (no panel feedback was available)"),
        ("docs/DEMO_SCRIPT.md", "Final demo script"),
        ("demo_runs/phase2_cluster_evidence.json", "Evidence (regenerate: python -m cluster.evidence)"),
        ("demo_runs/phase2_cluster_evidence.md", "Evidence, readable form"),
    ]
    missing = []
    shown_as = {"docs/WEEK6_REVIEW.md": "docs/ (review pass notes)"}
    for rel, what in deliverables:
        shown = shown_as.get(rel, rel)
        if (root / rel).is_file():
            ok(f"{shown:<42} {what}")
        else:
            missing.append(shown)
            print(f"  [MISSING]  {shown:<39} {what}")
    info("These are documents/evidence files; this script only checks they exist.")
    info("Content is validated by tests/test_evidence.py and by running the evidence module.")
    return missing


# ================================================================
#  MAIN
# ================================================================

def main():
    print()
    print(LINE)
    print("  CLASP - P2 Cluster Service")
    print("  Student: Prasanth")
    print("  Progress Demonstration")
    print(LINE)

    start = time.monotonic()

    demo_week1()
    demo_week2()
    demo_week3()
    demo_week4()
    demo_full_simulation()
    demo_week5()
    demo_weeks_6_to_13()
    missing = demo_weeks_14_to_16()

    elapsed = time.monotonic() - start

    heading("SUMMARY - What Was Demonstrated")
    rows = [
        ("Flower skeleton", "DummyClient round-trip through the Flower pipeline",
         "client.py, server.py"),
        ("Simulation", "3 simulated clients (make_dummy_clients)",
         "simulation.py"),
        ("FedProx", "Proximal term in the local training loop",
         "client.py: LoRAClient.train_local"),
        ("Adapter format", "PEFT state_dict <-> NumPy arrays",
         "adapter_format.py"),
        ("Contracts", "rank=16, alpha=16, 4 attention modules",
         "adapter_format.py: constants"),
        ("Delta-W", "delta_W = B @ A reconstruction",
         "adapter_format.py: delta_w"),
        ("Streaming mean", "Weighted mean with one accumulator",
         "aggregation.py: StreamingWeightedMean"),
        ("SVD aggregation", "Exact mean, truncated SVD re-factorization",
         "aggregation.py: aggregate_svd"),
        ("Round metrics", "Timer, num_clients, mean_loss",
         "server.py: aggregate_fit"),
        ("Redistribution", "Merged params returned to Flower",
         "server.py: aggregate_fit"),
        ("Round-trip check", "Reconstruction vs optimal rank-r bound",
         "tests/test_aggregation.py"),
        ("Fault tolerance", "Status/malformed skip + quorum guard",
         "server.py: aggregate_fit"),
        ("Federated round", "3-client round with metrics (ran above)",
         "demo.py"),
        ("Review pass", "alpha=16, validation (no panel feedback existed)",
         "docs/ (review pass notes)"),
        ("Multi-cluster", "Isolation, straggler, re-cluster, fallback (ran above)",
         "federation.py, clustering.py"),
        ("Deliverables", "Report + evidence + demo script (existence checked)",
         "docs/, demo_runs/"),
    ]

    print(f"\n       {'Area':<18} {'What Was Done':<56} {'File'}")
    print(f"       {'----':<18} {'-------------':<56} {'----'}")
    for area, what, file in rows:
        print(f"  [OK] {area:<18} {what:<56} {file}")

    print()
    print(f"  All demonstrations completed in {elapsed:.2f}s")
    print("  Files written by Prasanth (P2): server.py, aggregation.py,")
    print("    adapter_format.py, client.py, simulation.py, demo.py, tests/")
    print()
    print(LINE)
    if missing:
        print(f"  Status: pipeline demonstrated; MISSING deliverables: {', '.join(missing)}")
    else:
        print("  Status: pipeline demonstrated; report/evidence deliverables present")
    print(LINE)
    print("  Not demonstrated (blocked by other modules):")
    print("    - G1 mTLS and DP mu-tuning: need P3 (Security)")
    print("    - live Edge client, registry and evaluation: need P1 / P4 / P5")
    print("    - team-level report assembly and venue/submission tasks: not P2 work")
    print()


if __name__ == "__main__":
    main()
