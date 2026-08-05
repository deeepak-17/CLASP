    d"""Cluster server: Flower strategy with SVD LoRA aggregation (D2).

``SVDLoRAStrategy`` plugs the Week-3 aggregation core into Flower's FedAvg
skeleton: client updates arrive as flat ndarray lists (see
``LoRAAdapter.to_ndarrays`` for the wire order), are re-factorized through the
exact-average + truncated-SVD path, and the resulting cluster adapter is
broadcast back.

``build_server_app`` / ``start_grpc_server`` expose the same strategy through
Flower's new (ServerApp) and legacy (start_server) entry points; the in-process
driver in ``simulation.py`` uses the strategy math directly so tests need no
network or ray.
"""

from __future__ import annotations

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from services.cluster.adapter_format import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TARGET_MODULES,
    LoRAAdapter,
)
from services.cluster.aggregation import aggregate_naive, aggregate_svd


class SVDLoRAStrategy(FedAvg):
    """FedAvg skeleton with delta_W exact-average + SVD re-factorization."""

    def __init__(
        self,
        *args,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        aggregation: str = "svd",  # "svd" | "naive" (D2 ablation)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if aggregation not in ("svd", "naive"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.aggregation = aggregation
        self.round_log: list[dict[str, Scalar]] = []

    def _to_adapter(self, arrays) -> LoRAAdapter:
        return LoRAAdapter.from_ndarrays(
            arrays, rank=self.rank, alpha=self.alpha, target_modules=self.target_modules
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        adapters = (
            self._to_adapter(parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        )
        weights = [float(fit_res.num_examples) for _, fit_res in results]
        if self.aggregation == "svd":
            merged = aggregate_svd(adapters, weights, rank=self.rank)
        else:
            merged = aggregate_naive(adapters, weights)
        metrics: dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "aggregation": self.aggregation,
        }
        losses = [
            fit_res.metrics["loss"]
            for _, fit_res in results
            if fit_res.metrics and "loss" in fit_res.metrics
        ]
        if losses:
            metrics["mean_loss"] = float(sum(losses) / len(losses))
        self.round_log.append(metrics)
        return ndarrays_to_parameters(merged.to_ndarrays()), metrics


def build_strategy(
    initial_adapter: LoRAAdapter,
    aggregation: str = "svd",
    min_clients: int = 3,
) -> SVDLoRAStrategy:
    return SVDLoRAStrategy(
        rank=initial_adapter.rank,
        alpha=initial_adapter.alpha,
        target_modules=initial_adapter.target_modules,
        aggregation=aggregation,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_adapter.to_ndarrays()),
    )


def build_server_app(
    initial_adapter: LoRAAdapter,
    num_rounds: int = 3,
    aggregation: str = "svd",
    min_clients: int = 3,
):
    """Flower ServerApp (new API) around the SVD strategy — used by `flwr run`."""
    from flwr.server import ServerApp, ServerAppComponents

    strategy = build_strategy(initial_adapter, aggregation, min_clients)

    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=num_rounds)
        )

    return ServerApp(server_fn=server_fn)


def start_grpc_server(
    initial_adapter: LoRAAdapter,
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 3,
    aggregation: str = "svd",
    min_clients: int = 3,
):
    """Legacy blocking entry point (Week 1 skeleton); mTLS lands with P3 in W7."""
    import flwr

    return flwr.server.start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=build_strategy(initial_adapter, aggregation, min_clients),
    )
