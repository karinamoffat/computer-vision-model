import pytest
import torch

from modelSS import ARCHITECTURES, build_model, modelSS, modelSSResNet18, modelSSUNet


def make(arch):
    """All variants built offline so tests never hit the network."""
    return build_model(arch, num_classes=21, pretrained=False)


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_output_shape_matches_input_resolution(arch):
    model = make(arch)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 21, 256, 256)


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_num_classes_is_respected(arch):
    model = build_model(arch, num_classes=5, pretrained=False)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))

    assert out.shape == (1, 5, 64, 64)


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_backward_pass_reaches_every_parameter(arch):
    """A skip wired to the wrong stage often shows up as a parameter with no grad."""
    model = make(arch)
    out = model(torch.randn(1, 3, 64, 64))
    out.sum().backward()

    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert missing == [], f'no gradient reached: {missing}'


def test_build_model_returns_the_right_classes():
    assert isinstance(make('baseline'), modelSS)
    assert isinstance(make('unet'), modelSSUNet)
    assert isinstance(make('resnet18'), modelSSResNet18)


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match='unknown arch'):
        build_model('not-an-arch')


def test_unet_matches_baseline_encoder_widths():
    """Skips should be the only difference, so encoder capacity stays comparable."""
    base, unet = make('baseline'), make('unet')

    for layer in ('conv1', 'conv2', 'conv3', 'bottleneck'):
        assert getattr(base, layer).weight.shape == getattr(unet, layer).weight.shape


def test_baseline_checkpoint_still_loads():
    """The baseline class is unchanged, so earlier tiers' checkpoints stay loadable."""
    a, b = make('baseline'), make('baseline')
    b.load_state_dict(a.state_dict())
