"""Regression tests for the P1 review of #19 (findings 1-10 at dbaa4eb)."""
from __future__ import annotations

import dataclasses
import datetime
import ssl

import numpy as np
import pytest

from contracts import AdapterKind, AdapterUpload, LoRAHyperParams
from starlette.requests import Request
from security.mtls.ca import create_ca, save_pem
from security.mtls.certs import generate_service_cert, verify_cert
from security.mtls.context import client_ssl_context, server_ssl_context
from security.validation.adapter_validator import (
    AdapterValidationConfig,
    detect_poisoning,
    validate_adapter_integrity,
    validate_adapter_metadata,
)

torch = pytest.importorskip("torch")
pytest.importorskip("opacus")
from opacus.validators import ModuleValidator  # noqa: E402 — after the importorskip guard
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from security.dp.accountant import DEFAULT_ACCOUNTANT, PrivacyAccountant  # noqa: E402
from security.dp.config import DPConfig  # noqa: E402
from security.dp.engine import _rebuild_optimizer, get_privacy_engine, make_private  # noqa: E402


# -- 1. one accountant: a calibrated run passes PrivacyAccountant ------------ #
def test_calibrated_run_passes_the_library_accountant():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 2))
    loader = DataLoader(TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,))),
                        batch_size=8)
    cfg = DPConfig(enabled=True, max_grad_norm=1.0, target_epsilon=8.0, delta=1e-5)
    epochs = 3
    model_dp, opt_dp, loader_dp = make_private(
        model, torch.optim.SGD(model.parameters(), lr=0.1), loader, cfg, epochs=epochs)
    steps = 0
    for _ in range(epochs):
        for x, y in loader_dp:
            if len(x) == 0:
                continue
            opt_dp.zero_grad()
            nn.functional.cross_entropy(model_dp(x), y).backward()
            opt_dp.step()
            steps += 1
    engine = get_privacy_engine(model_dp)
    assert engine.accountant.mechanism() == DEFAULT_ACCOUNTANT == "prv"

    tracked = PrivacyAccountant(
        DPConfig(noise_multiplier=opt_dp.noise_multiplier, target_epsilon=8.0, delta=1e-5),
        sample_rate=1 / len(loader))
    eps = tracked.step(steps)  # raised BudgetExhaustedError with the RDP accountant
    assert eps <= 8.0
    assert eps == pytest.approx(engine.get_epsilon(1e-5), rel=0.05)


def test_reviewers_run_is_inside_the_budget_and_rdp_stays_available():
    cfg = DPConfig(noise_multiplier=0.6519, target_epsilon=8.0, delta=1e-5)
    assert PrivacyAccountant(cfg, sample_rate=0.0769).step(26) <= 8.0
    rdp = PrivacyAccountant(cfg, sample_rate=0.0769, accountant="rdp")
    assert rdp.compute_epsilon(26, 0.0769) > 8.0  # the looser bound, on request


# -- 2. poisoning detection works at D1's cluster sizes ---------------------- #
def _adapters(norms):
    return [{"w": np.full((4,), n / 2.0, dtype=np.float32)} for n in norms]  # ||w|| == n


@pytest.mark.parametrize("n", [3, 6])
def test_a_1000x_client_is_flagged_at_n(n):
    norms = [1.0 + 0.05 * i for i in range(n - 1)] + [1000.0]
    assert detect_poisoning(_adapters(norms)) == [n - 1]


def test_a_vanishing_client_is_flagged_too():
    assert detect_poisoning(_adapters([1.0, 1.1, 0.001])) == [2]


def test_ordinary_spread_is_not_flagged():
    assert detect_poisoning(_adapters([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])) == []
    assert detect_poisoning(_adapters([1.0, 1.05, 1.5])) == []


def test_too_few_adapters_to_compare():
    assert detect_poisoning(_adapters([1.0, 1000.0])) == []


# -- 3. the optimizer keeps its param groups after ModuleValidator.fix ------- #
def _bn_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    for p in model[0].parameters():
        p.requires_grad_(False)  # frozen layer
    opt = torch.optim.SGD([{"params": model[2].parameters(), "lr": 0.5},
                           {"params": model[1].parameters(), "lr": 0.01}], lr=0.1)
    return model, opt


def test_rebuild_keeps_groups_and_leaves_frozen_params_out():
    model, opt = _bn_model()
    fixed = ModuleValidator.fix(model)
    rebuilt = _rebuild_optimizer(opt, model, fixed)
    assert [g["lr"] for g in rebuilt.param_groups] == [0.5, 0.01]
    held = {id(p) for g in rebuilt.param_groups for p in g["params"]}
    fixed_params = dict(fixed.named_parameters())
    assert held == {id(fixed_params[n]) for n in ("2.weight", "2.bias", "1.weight", "1.bias")}
    assert id(fixed_params["0.weight"]) not in held


def test_make_private_on_a_fixed_model_trains_with_the_groups():
    model, opt = _bn_model()
    loader = DataLoader(TensorDataset(torch.randn(32, 4), torch.randint(0, 2, (32,))),
                        batch_size=8)
    model_dp, opt_dp, loader_dp = make_private(
        model, opt, loader, DPConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0))
    assert [g["lr"] for g in opt_dp.param_groups] == [0.5, 0.01]
    x, y = next(b for b in loader_dp if len(b[0]))
    nn.functional.cross_entropy(model_dp(x), y).backward()
    opt_dp.step()


def test_rebuild_refuses_a_parameter_it_cannot_match():
    model, opt = _bn_model()
    with pytest.raises(ValueError, match="no counterpart"):
        _rebuild_optimizer(opt, model, nn.Sequential(nn.Linear(4, 2)))


# -- 4. the metadata validator takes the contract's own upload shape --------- #
def _upload(**over):
    fields = dict(client_id="flask", cluster_id="web", kind=AdapterKind.CLIENT,
                  hparams=LoRAHyperParams(rank=16), num_train_samples=200, round=2)
    return AdapterUpload(**{**fields, **over})


def test_contract_upload_validates_as_instance_and_json():
    assert validate_adapter_metadata(_upload()) == []
    assert validate_adapter_metadata(dataclasses.asdict(_upload())) == []


def test_contract_upload_errors_name_contract_fields():
    bad = {**dataclasses.asdict(_upload()), "round": "two"}
    bad["hparams"] = {**bad["hparams"], "rank": "sixteen"}
    errors = validate_adapter_metadata(bad)
    assert "round must be an integer" in errors
    assert "hparams.rank must be an integer" in errors


# -- 5. max_rank is enforced (D8: 16) ---------------------------------------- #
def test_rank_above_max_rank_is_rejected():
    big = {"layers.0.q_proj.lora_A.weight": np.zeros((512, 8), dtype=np.float32),
           "layers.0.q_proj.lora_B.weight": np.zeros((8, 512), dtype=np.float32)}
    result = validate_adapter_integrity(big, AdapterValidationConfig(max_rank=128))
    assert not result.valid and sum("rank 512" in e for e in result.errors) == 2


def test_default_max_rank_is_d8s_cap():
    ok = {"q.lora_A.weight": np.zeros((16, 8), dtype=np.float32),
          "q.lora_B.weight": np.zeros((8, 16), dtype=np.float32)}
    assert validate_adapter_integrity(ok).valid
    over = {"q.lora_A.weight": np.zeros((32, 8), dtype=np.float32)}
    assert not validate_adapter_integrity(over).valid


# -- 6, 9. request body limit ------------------------------------------------ #
@pytest.fixture
def body_app():
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from security.middleware.request_validator import RequestValidationMiddleware

    app = FastAPI()
    app.add_middleware(RequestValidationMiddleware, max_body_bytes=1000)

    async def echo(request):
        return {"received": len(await request.body())}

    # Annotated after definition: with postponed annotations FastAPI cannot
    # resolve a Request imported inside this fixture.
    echo.__annotations__ = {"request": Request, "return": dict}
    app.post("/echo")(echo)

    return TestClient(app)


def _chunks(total, size=100):
    for _ in range(total // size):
        yield b"x" * size


def test_chunked_body_under_the_limit_reaches_the_endpoint(body_app):
    r = body_app.post("/echo", content=_chunks(900))
    assert r.status_code == 200 and r.json() == {"received": 900}


def test_chunked_body_over_the_limit_is_refused(body_app):
    assert body_app.post("/echo", content=_chunks(5000)).status_code == 413


def test_malformed_content_length_is_a_400(body_app):
    for value in ("abc", "-5"):
        r = body_app.post("/echo", content=b"hi", headers={"content-length": value})
        assert r.status_code == 400


# -- 8. verify_cert checks issuer and validity window ------------------------ #
def _signed(ca_key, ca_cert, *, issuer=None, start_days=-1, end_days=30):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "svc")]))
            .issuer_name(issuer or ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now + datetime.timedelta(days=start_days))
            .not_valid_after(now + datetime.timedelta(days=end_days))
            .sign(ca_key, hashes.SHA256()))


@pytest.fixture(scope="module")
def ca():
    return create_ca(common_name="Review CA", key_size=2048)


def test_verify_cert_rejects_expired_future_and_wrong_issuer(ca):
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    ca_key, ca_cert = ca
    assert verify_cert(_signed(ca_key, ca_cert), ca_cert) is True
    assert verify_cert(_signed(ca_key, ca_cert, start_days=-30, end_days=-1), ca_cert) is False
    assert verify_cert(_signed(ca_key, ca_cert, start_days=1, end_days=30), ca_cert) is False
    other = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Other CA")])
    assert verify_cert(_signed(ca_key, ca_cert, issuer=other), ca_cert) is False


# -- 10. one TLS policy: 1.3 on both sides ----------------------------------- #
def test_contexts_require_tls_1_3(ca, tmp_path):
    ca_key, ca_cert = ca
    save_pem(ca_key, ca_cert, tmp_path, name="ca")
    key, cert = generate_service_cert(ca_key, ca_cert, "svc", sans=["localhost"])
    save_pem(key, cert, tmp_path, name="svc")
    args = (tmp_path / "svc_cert.pem", tmp_path / "svc_key.pem", tmp_path / "ca_cert.pem")
    for ctx in (server_ssl_context(*args), client_ssl_context(*args)):
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_3
