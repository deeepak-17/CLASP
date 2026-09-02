import pytest
import torch


class TestSetup:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs a CUDA GPU; CI runners are CPU-only",
    )
    def test_cuda_available(self):
        # Still asserts on a GPU box (P1's local check) — skipped, not failed, in CI.
        print(torch.version.cuda)
        assert torch.cuda.is_available()
