"""``edge.webui``: the edge laptop's page, driven over HTTP against the real cluster."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from edge import webui
from tests.test_upload import _adapter, _register, cluster_url  # noqa: F401 - fixture

server = pytest.importorskip("cluster.server")


@pytest.fixture(autouse=True)
def _clean():
    server._reset_clusters()
    yield
    server._reset_clusters()


@pytest.fixture
def page_url(cluster_url, tmp_path):  # noqa: F811 - the imported fixture
    _adapter(tmp_path / "round2_d3", "client-flask", seed=1, epsilon=7.99)
    _adapter(tmp_path / "round1", "client-flask", seed=2)
    _adapter(tmp_path / "round1", "client-numpy", seed=3, cluster="scientific")
    page = webui.EdgePage(cluster_url=cluster_url, client_id="client-flask", adapters_root=tmp_path)
    httpd = webui.serve(page, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", page
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_job(url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = _get(f"{url}/api/job")
        if not job["running"]:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_page_lists_only_this_clients_adapters(page_url):
    url, _ = page_url
    with urllib.request.urlopen(url, timeout=10) as r:
        assert b"CLASP Edge" in r.read()
    adapters = _get(f"{url}/api/adapters")["adapters"]
    assert {a["id"] for a in adapters} == {"round1/client-flask", "round2_d3/client-flask"}
    d3 = next(a for a in adapters if a["set"] == "round2_d3")
    assert d3["epsilon"] == 7.99 and d3["cluster"] == "web" and "path" not in d3


def test_status_then_upload_shows_the_client_arrive(page_url, cluster_url):  # noqa: F811
    url, _ = page_url
    status = _get(f"{url}/api/status")
    assert status["reachable"] and "web" not in status["clusters"] and status["my_cluster"] is None
    _register(cluster_url, "web", ["client-flask", "client-requests"])
    assert _get(f"{url}/api/status")["my_cluster"] == "web"
    code, body = _post(f"{url}/api/upload", {"adapter": "round2_d3/client-flask"})
    assert code == 202, body
    job = _wait_job(url)
    assert job["error"] is None and job["result"]["cluster_id"] == "web"
    assert job["result"]["epsilon"] == 7.99
    web = _get(f"{url}/api/status")["clusters"]["web"]
    assert web["uploaded_clients"] == ["client-flask"]


def test_refusals_reach_the_page(page_url):
    url, _ = page_url
    code, _ = _post(f"{url}/api/upload", {"adapter": "../../etc/passwd"})
    assert code == 404
    code, _ = _post(f"{url}/api/upload", {"adapter": "round1/client-flask"})
    assert code == 202
    job = _wait_job(url)
    assert "not registered" in job["error"]  # nobody registered 'web' on the cluster
    code, body = _post(f"{url}/api/train", {"client": "web/client-flask", "max_steps": 0})
    assert code == 422 and "between 1 and 2000" in body["detail"]
    code, body = _post(f"{url}/api/train", {"client": "../x", "max_steps": 5})
    assert code == 422


def test_unreachable_cluster_is_reported(tmp_path):
    page = webui.EdgePage(cluster_url="http://127.0.0.1:9", client_id=None, adapters_root=tmp_path)
    s = page.status()
    assert not s["reachable"] and "firewall" in s["hint"]


def test_train_command(tmp_path):
    page = webui.EdgePage(cluster_url="http://x", client_id="client-flask", adapters_root=tmp_path,
                          corpus_root=tmp_path / "corpus", python="py")
    cmd = page.train_command(client="web/client-flask", max_steps=40, dp=True,
                             cluster_adapter="c/web")
    assert cmd[:6] == ["py", "-m", "edge.train_client", "--client", "web/client-flask", "--max-steps"]
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path / webui.LIVE_SET)
    assert "--dp" in cmd and cmd[cmd.index("--cluster-adapter") + 1] == "c/web"
