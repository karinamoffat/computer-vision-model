import torch

from modelSS import modelSS


def test_forward_pass_output_shape():
    model = modelSS(num_classes=21)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 21, 256, 256)
