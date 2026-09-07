"""Edge-side State Registry client (Integration Sprint · B4, seams C1 and C2).

    C1  GET  /adapters/{name}/active            metadata for the live version
        GET  /adapters/{name}/versions/{v}/file the safetensors payload
        -> verify sha256 -> materialize a PEFT directory edge.merge can load

    C2  POST /adapters/{name}/promote           EvalResult in, D5 decision out

Two things this deliberately does NOT do.

**It does not invent metadata.** ``adapter_config.json`` is rebuilt from what
the registry actually recorded — ``hparams.rank`` -> ``r``,
``hparams.lora_alpha`` -> ``lora_alpha``, ``hparams.target_modules`` -> the
target module list — plus the PEFT config the producer embedded in the
safetensors ``__metadata__`` header when it serialized the blob (see
``edge.wire.serialize`` / ``cluster.server._peft_bytes``). Fields present in
both are CROSS-CHECKED, and a disagreement raises rather than being papered
over. If a payload carries no embedded config, the caller must name the base
model explicitly; the client will not guess one. Every field's provenance is
written to ``materialize_manifest.json`` beside the adapter.

**It does not trust the bytes.** The registry records a sha256 per payload
(``AdapterMetadata.sha256``). Every download is hashed and compared before a
single tensor is written to disk; a mismatch raises ``ChecksumMismatch`` and
leaves nothing behind, because a silently corrupt adapter produces a wrong
composite rather than an error.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import requests

DEFAULT_REGISTRY_URL = "http://localhost:8004"
DEFAULT_TIMEOUT = 120.0

#: PEFT structural fields that decide what the stored tensors MEAN. The
#: registry does not record them (contracts v1.0 LoRAHyperParams has no room
#: for them), so a payload with no embedded config falls back to these — which
#: are exactly the values ``edge.merge.STRUCTURAL_FIELDS`` requires two
#: adapters to agree on before they may be composed.
PEFT_STRUCTURAL_DEFAULTS: Dict[str, Any] = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "use_rslora": False,
    "use_dora": False,
    "fan_in_fan_out": False,
    "lora_bias": False,
    "bias": "none",
    "inference_mode": True,
    "rank_pattern": {},
    "alpha_pattern": {},
    "modules_to_save": None,
    "init_lora_weights": True,
}


class RegistryError(RuntimeError):
    """The registry answered, but not with what the caller needs."""


class ChecksumMismatch(RegistryError):
    """Downloaded payload does not hash to the sha256 the registry recorded."""


class MetadataConflict(RegistryError):
    """Registry hyperparameters and the payload's embedded config disagree."""


class RegistryClient:
    """Thin HTTP client over the eight registry endpoints Edge needs."""

    def __init__(self, base_url: str = DEFAULT_REGISTRY_URL,
                 timeout: float = DEFAULT_TIMEOUT, session=None) -> None:
        """``session`` overrides the HTTP transport.

        Anything with requests' ``get``/``post``/``close`` will do — in
        particular FastAPI's ``TestClient``, which lets the CPU-only CI test
        drive real seam C1/C2 requests through the registry app in-process,
        with no socket and no running service. Production callers leave it
        unset and get a plain ``requests.Session``.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._injected = session is not None
        self._session = session if session is not None else requests.Session()

    # -- plumbing ---------------------------------------------------------- #
    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @property
    def _timeout(self) -> dict:
        """An injected transport (TestClient) manages its own timeouts and warns
        if one is passed; a real requests.Session needs ours."""
        return {} if self._injected else {"timeout": self.timeout}

    def _get(self, path: str, *, raw: bool = False):
        resp = self._session.get(self._url(path), **self._timeout)
        if resp.status_code >= 400:
            raise RegistryError(f"GET {path} -> {resp.status_code}: {resp.text[:400]}")
        return resp.content if raw else resp.json()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reads -------------------------------------------------------------- #
    def healthz(self) -> Dict:
        return self._get("/healthz")

    def list_adapters(self) -> list[str]:
        return self._get("/adapters")["adapters"]

    def list_versions(self, name: str) -> Dict:
        return self._get(f"/adapters/{name}/versions")

    def get_version(self, name: str, version: int) -> Dict:
        return self._get(f"/adapters/{name}/versions/{version}")

    def get_active(self, name: str) -> Dict:
        """C1, half one: the metadata of whichever version is currently live."""
        return self._get(f"/adapters/{name}/active")

    def download(self, name: str, version: int) -> bytes:
        """C1, half two: the raw safetensors payload for one version."""
        return self._get(f"/adapters/{name}/versions/{version}/file", raw=True)

    def list_promotions(self, name: str) -> list[Dict]:
        return self._get(f"/adapters/{name}/promotions")["decisions"]

    # -- writes ------------------------------------------------------------- #
    def save_version(self, name: str, payload: bytes, meta: Dict) -> Dict:
        """POST a new immutable version (seam B, when the orchestrator drives it).

        The cluster service publishes its own snapshots via
        ``POST /adapters/{cluster_id}/publish``; this is the same call from the
        Edge side, used for edge-produced adapters (composites) and as the
        fallback path if the cluster cannot reach the registry itself.
        """
        resp = self._session.post(
            self._url(f"/adapters/{name}/versions"),
            files={"file": (f"{name}.safetensors", payload, "application/octet-stream")},
            data={"meta": json.dumps(meta)},
            **self._timeout,
        )
        if resp.status_code >= 400:
            raise RegistryError(
                f"POST /adapters/{name}/versions -> {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def promote(self, name: str, eval_result: Dict,
                baseline_guard: Sequence[Dict] = ()) -> Dict:
        """C2: hand the registry an EvalResult and get the D5 decision back.

        The rule itself lives in ``registry.promotion.decide`` and is not
        second-guessed here: whatever comes back — PROMOTE or ROLLBACK — is the
        registry's decision, recorded in its own audit trail.
        """
        body = {"eval": eval_result, "baseline_guard": list(baseline_guard)}
        resp = self._session.post(
            self._url(f"/adapters/{name}/promote"), json=body, **self._timeout)
        if resp.status_code >= 400:
            raise RegistryError(
                f"POST /adapters/{name}/promote -> {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    # -- C1: materialize a loadable PEFT directory --------------------------- #
    def fetch_verified(self, name: str, version: Optional[int] = None
                       ) -> Tuple[bytes, Dict]:
        """Download a version's payload and check it against the recorded sha256.

        Returns (payload, metadata). Raises :class:`ChecksumMismatch` on a
        digest disagreement — loudly, before anything touches the filesystem.
        """
        meta = self.get_active(name) if version is None else self.get_version(name, version)
        version = meta["ref"]["version"]
        payload = self.download(name, version)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != meta["sha256"]:
            raise ChecksumMismatch(
                f"{name} v{version}: registry recorded sha256={meta['sha256']}, "
                f"downloaded payload hashes to {digest} "
                f"({len(payload)} bytes vs {meta['num_bytes']} recorded)")
        if len(payload) != meta["num_bytes"]:
            raise ChecksumMismatch(
                f"{name} v{version}: payload is {len(payload)} bytes, registry "
                f"recorded {meta['num_bytes']}")
        return payload, meta

    def materialize(self, name: str, out_dir: Path | str,
                    version: Optional[int] = None,
                    base_model: Optional[str] = None,
                    overwrite: bool = True) -> Dict:
        """C1 end to end: pull the active version and write a PEFT directory.

        Produces ``adapter_config.json`` + ``adapter_model.safetensors`` —
        exactly the two files ``edge.merge.load_adapter`` reads — plus a
        ``materialize_manifest.json`` recording where every config field came
        from and both sha256 digests.
        """
        from edge import wire

        payload, meta = self.fetch_verified(name, version)
        version = meta["ref"]["version"]
        state_dict, embedded = wire.deserialize(payload)
        cfg, provenance = self._build_config(meta, embedded, base_model)

        out_dir = Path(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            if not overwrite:
                raise RegistryError(f"{out_dir} already exists and overwrite=False")
            shutil.rmtree(out_dir)
        # Stage then rename, so an interrupted materialize never leaves a
        # half-written adapter directory that the merge path would happily load.
        staging = Path(tempfile.mkdtemp(prefix=f".tmp-{name}-v{version}-",
                                        dir=out_dir.parent))
        try:
            wire.save_peft_dir(staging, wire.to_peft_keys(state_dict), cfg)
            manifest = {
                "task": "Integration Sprint C1 — registry -> edge materialization",
                "registry_url": self.base_url,
                "adapter": name,
                "version": version,
                "kind": meta["ref"]["kind"],
                "cluster_id": meta["ref"].get("cluster_id"),
                "aggregation": meta.get("aggregation"),
                "round": meta.get("round"),
                "source_clients": meta.get("source_clients", []),
                "sha256_recorded": meta["sha256"],
                "sha256_verified": hashlib.sha256(payload).hexdigest(),
                "num_bytes": meta["num_bytes"],
                "n_tensors": len(state_dict),
                "adapter_config_provenance": provenance,
                "created_at": meta.get("created_at"),
            }
            (staging / "materialize_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            staging.replace(out_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        manifest["path"] = str(out_dir)
        return manifest

    # -- config reconstruction ---------------------------------------------- #
    @staticmethod
    def _build_config(meta: Dict, embedded: Optional[Dict],
                      base_model: Optional[str]) -> Tuple[Dict, Dict[str, str]]:
        """adapter_config.json from registry metadata + the embedded config.

        Registry hyperparameters win on the fields the registry records, and
        any disagreement with the embedded config is an error, not a
        preference: the two describe the same tensors, so if they differ one of
        them is lying about what is in the blob.
        """
        hp = meta["hparams"]
        provenance: Dict[str, str] = {}
        cfg: Dict[str, Any] = dict(PEFT_STRUCTURAL_DEFAULTS)
        for key in cfg:
            provenance[key] = "edge.registry_client.PEFT_STRUCTURAL_DEFAULTS"

        if embedded:
            for key, value in embedded.items():
                cfg[key] = value
                provenance[key] = "safetensors __metadata__ (producer-embedded)"

        checks = {
            "r": ("rank", hp["rank"]),
            "lora_alpha": ("lora_alpha", hp["lora_alpha"]),
            "lora_dropout": ("dropout", hp["dropout"]),
        }
        for cfg_key, (hp_key, value) in checks.items():
            if embedded and cfg_key in embedded and float(embedded[cfg_key]) != float(value):
                raise MetadataConflict(
                    f"registry hparams.{hp_key}={value} but the payload's embedded "
                    f"adapter_config says {cfg_key}={embedded[cfg_key]}")
            cfg[cfg_key] = value
            provenance[cfg_key] = f"registry metadata hparams.{hp_key}"

        registry_modules = set(hp["target_modules"])
        if embedded and set(embedded.get("target_modules") or ()) != registry_modules:
            raise MetadataConflict(
                f"registry hparams.target_modules={sorted(registry_modules)} but the "
                f"payload's embedded adapter_config says "
                f"{sorted(embedded.get('target_modules') or ())}")
        cfg["target_modules"] = sorted(registry_modules)
        provenance["target_modules"] = "registry metadata hparams.target_modules"

        # The one field contracts v1.0 has nowhere to record. Take it from the
        # embedded config if the producer left one; otherwise the caller must
        # say. Never guessed.
        if base_model is not None:
            if (embedded and embedded.get("base_model_name_or_path")
                    and embedded["base_model_name_or_path"] != base_model):
                raise MetadataConflict(
                    f"caller passed base_model={base_model!r} but the payload was "
                    f"produced against {embedded['base_model_name_or_path']!r}")
            cfg["base_model_name_or_path"] = base_model
            provenance["base_model_name_or_path"] = "caller-supplied"
        elif not cfg.get("base_model_name_or_path"):
            raise MetadataConflict(
                "payload carries no embedded adapter_config and the registry does "
                "not record a base model (contracts v1.0 LoRAHyperParams has no "
                "such field) — pass base_model= explicitly rather than guessing")
        return cfg, provenance


# --------------------------------------------------------------------------- #
# CLI — `python -m edge.registry_client pull cluster-web ./out`
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Edge <-> State Registry client (C1/C2)")
    ap.add_argument("command", choices=["health", "list", "versions", "pull", "promotions"])
    ap.add_argument("name", nargs="?", help="adapter name")
    ap.add_argument("out", nargs="?", help="destination directory for `pull`")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY_URL)
    ap.add_argument("--version", type=int, default=None, help="default: the active one")
    ap.add_argument("--base-model", default=None)
    args = ap.parse_args()

    with RegistryClient(args.registry) as rc:
        if args.command == "health":
            print(json.dumps(rc.healthz(), indent=2))
        elif args.command == "list":
            print("\n".join(rc.list_adapters()) or "(no adapters)")
        elif args.command == "versions":
            print(json.dumps(rc.list_versions(args.name), indent=2))
        elif args.command == "promotions":
            print(json.dumps(rc.list_promotions(args.name), indent=2))
        else:
            if not (args.name and args.out):
                raise SystemExit("pull needs NAME and OUT")
            manifest = rc.materialize(args.name, args.out, version=args.version,
                                      base_model=args.base_model)
            print(f"{args.name} v{manifest['version']} -> {manifest['path']}")
            print(f"  sha256 verified : {manifest['sha256_verified'][:16]}...")
            print(f"  source clients  : {', '.join(manifest['source_clients']) or '—'}")
            print(f"  aggregation     : {manifest['aggregation']}")
            print(f"  tensors         : {manifest['n_tensors']}")


if __name__ == "__main__":
    main()
