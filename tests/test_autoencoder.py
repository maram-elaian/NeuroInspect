import torch
from src.models.autoencoder import ConvAE


def test_shapes():
    m = ConvAE().eval()
    x = torch.randn(2, 3, 224, 224)
    assert m(x).shape == x.shape
    assert m.encoder(x).shape == (2, 64, 7, 7)
