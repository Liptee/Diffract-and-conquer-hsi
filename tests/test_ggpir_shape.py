from diffract_conquer_hsi.models.ggpir import GGPIR


def test_ggpir_forward_shape():
    import torch

    m = GGPIR(num_bands=31, num_gaussian_outputs=7)
    x = torch.randn(2, 31, 16, 16)
    y = m(x)
    assert y.shape == x.shape
