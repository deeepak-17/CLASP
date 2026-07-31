import torch

class TestSetup:
    def test_cuda_available(self):
        print(torch.version.cuda)
        assert torch.cuda.is_available()
