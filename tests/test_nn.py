"""Tests for neural network models."""

import torch
import pytest

from colorspace.config import TrainConfig
from colorspace.nn.inn import ColorINN
from colorspace.nn.mlp import ColorMLP
from colorspace.nn.losses import STRESSLoss, HueLinearityLoss, RoundTripLoss, D4RegularizationLoss, HKEffectLoss, CombinedLoss


@pytest.fixture
def small_cfg():
    cfg = TrainConfig()
    cfg.inn_coupling_blocks = 4
    cfg.inn_subnet_hidden = 32
    cfg.mlp_hidden = 32
    cfg.mlp_layers = 2
    return cfg


class TestColorINN:
    def test_forward_shape(self, small_cfg):
        model = ColorINN(small_cfg)
        x = torch.randn(10, 3)
        out = model(x)
        assert out.shape == (10, 3)

    def test_inverse_shape(self, small_cfg):
        model = ColorINN(small_cfg)
        p = torch.randn(10, 3)
        out = model.inverse(p)
        assert out.shape == (10, 3)

    def test_round_trip(self, small_cfg):
        model = ColorINN(small_cfg)
        model.eval()
        x = torch.randn(50, 3)
        with torch.no_grad():
            z_full, _ = model.forward_full(x)
            x_rec, _ = model.inverse_full(z_full)
        err = (x_rec[:, :3] - x).abs().max().item()
        assert err < 1e-5, f"INN round-trip error: {err:.2e}"

    def test_forward_full_shape_3d(self, small_cfg):
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        x = torch.randn(10, 3)
        z_full, log_jac = model.forward_full(x)
        assert z_full.shape == (10, 3)
        assert log_jac.shape == (10,)

    def test_forward_full_shape_4d(self, small_cfg):
        small_cfg.inn_pad_dim = 1
        model = ColorINN(small_cfg)
        x = torch.randn(10, 3)
        z_full, log_jac = model.forward_full(x)
        assert z_full.shape == (10, 4)
        assert log_jac.shape == (10,)

    def test_d4_accessible(self, small_cfg):
        """d4 dimension is extractable from 4D forward_full output."""
        small_cfg.inn_pad_dim = 1
        model = ColorINN(small_cfg)
        x = torch.randn(10, 3)
        z_full, _ = model.forward_full(x)
        d4 = z_full[:, 3]
        assert d4.shape == (10,)

    def test_gradient_flow(self, small_cfg):
        model = ColorINN(small_cfg)
        x = torch.randn(5, 3, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_d4_gradient_flow(self, small_cfg):
        """d4 regularization loss propagates gradients."""
        small_cfg.inn_pad_dim = 1
        model = ColorINN(small_cfg)
        x = torch.randn(5, 3)
        z_full, _ = model.forward_full(x)
        d4 = z_full[:, 3]
        loss = torch.mean(d4 ** 2)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "d4 loss did not produce gradients"


class TestColorMLP:
    def test_forward_shape(self, small_cfg):
        model = ColorMLP(small_cfg)
        x = torch.randn(10, 3)
        assert model(x).shape == (10, 3)

    def test_inverse_shape(self, small_cfg):
        model = ColorMLP(small_cfg)
        p = torch.randn(10, 3)
        assert model.inverse(p).shape == (10, 3)


class TestLosses:
    def test_stress_loss_perfect(self):
        loss_fn = STRESSLoss()
        p1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        p2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        # DE = [0, sqrt(2)]
        DV = torch.tensor([0.0, 1.414], dtype=torch.float32)
        s = loss_fn(p1, p2, DV)
        assert s.item() >= 0

    def test_stress_loss_differentiable(self):
        loss_fn = STRESSLoss()
        p1 = torch.randn(10, 3, requires_grad=True)
        p2 = torch.randn(10, 3, requires_grad=True)
        DV = torch.rand(10)
        s = loss_fn(p1, p2, DV)
        s.backward()
        assert p1.grad is not None

    def test_hue_linearity_loss(self):
        loss_fn = HueLinearityLoss()
        # Perfect line in a-b plane
        perceptual = torch.zeros(10, 3)
        perceptual[:, 1] = torch.linspace(0, 1, 10)  # a varies
        perceptual[:, 2] = torch.linspace(0, 1, 10)  # b = a (45° line)
        hue_idx = torch.zeros(10, dtype=torch.long)
        l = loss_fn(perceptual, hue_idx, n_hues=1)
        assert l.item() == pytest.approx(0.0, abs=1e-5)

    def test_round_trip_loss(self):
        loss_fn = RoundTripLoss()
        x = torch.randn(10, 3)
        l = loss_fn(x, x)
        assert l.item() == pytest.approx(0.0, abs=1e-10)

    def test_d4_reg_loss_zero(self):
        loss_fn = D4RegularizationLoss()
        d4 = torch.zeros(10)
        assert loss_fn(d4).item() == pytest.approx(0.0, abs=1e-10)

    def test_d4_reg_loss_positive(self):
        loss_fn = D4RegularizationLoss()
        d4 = torch.tensor([1.0, -1.0, 2.0])
        mean_sq = (1.0 + 1.0 + 4.0) / 3.0
        max_sq = 4.0
        expected = mean_sq + max_sq
        assert loss_fn(d4).item() == pytest.approx(expected, abs=1e-5)

    def test_d4_reg_loss_differentiable(self):
        loss_fn = D4RegularizationLoss()
        d4 = torch.randn(10, requires_grad=True)
        l = loss_fn(d4)
        l.backward()
        assert d4.grad is not None


class TestCombinedLossD4:
    def test_combined_loss_with_d4(self, small_cfg):
        """CombinedLoss includes d4 regularization for 4D INN models."""
        small_cfg.inn_pad_dim = 1
        model = ColorINN(small_cfg)
        criterion = CombinedLoss(delta_d4=1.0, warmup_epochs=0)
        XYZ_1 = torch.randn(20, 3)
        XYZ_2 = torch.randn(20, 3)
        DV = torch.rand(20) + 0.1
        losses = criterion(model, XYZ_1, XYZ_2, DV)
        assert "d4_reg" in losses
        assert losses["d4_reg"].item() >= 0
        assert losses["total"].requires_grad

    def test_combined_loss_d4_zero_weight(self, small_cfg):
        """When delta_d4=0, d4_reg loss should be 0."""
        small_cfg.inn_pad_dim = 1
        model = ColorINN(small_cfg)
        criterion = CombinedLoss(delta_d4=0.0, warmup_epochs=0)
        XYZ_1 = torch.randn(20, 3)
        XYZ_2 = torch.randn(20, 3)
        DV = torch.rand(20) + 0.1
        losses = criterion(model, XYZ_1, XYZ_2, DV)
        assert losses["d4_reg"].item() == pytest.approx(0.0, abs=1e-10)


class TestHKEffectLoss:
    def test_hk_loss_output_shape(self, small_cfg):
        """HKEffectLoss returns a scalar."""
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        loss_fn = HKEffectLoss(n_samples=16)
        loss = loss_fn(model)
        assert loss.dim() == 0  # scalar

    def test_hk_loss_non_negative(self, small_cfg):
        """HKEffectLoss is always >= 0 (margin loss)."""
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        loss_fn = HKEffectLoss(n_samples=32)
        loss = loss_fn(model)
        assert loss.item() >= 0.0

    def test_hk_loss_differentiable(self, small_cfg):
        """HKEffectLoss propagates gradients to model parameters."""
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        loss_fn = HKEffectLoss(n_samples=16)
        loss = loss_fn(model)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "HKEffectLoss did not produce gradients"

    def test_hk_loss_generates_valid_xyz(self, small_cfg):
        """Generated XYZ pairs should have positive Y values."""
        loss_fn = HKEffectLoss(n_samples=32)
        XYZ_a, XYZ_c = loss_fn._generate_hk_pairs(torch.device("cpu"))
        assert (XYZ_a[:, 1] > 0).all(), "Achromatic Y should be positive"
        assert (XYZ_c[:, 1] > 0).all(), "Chromatic Y should be positive"
        # Same Y for both
        assert torch.allclose(XYZ_a[:, 1], XYZ_c[:, 1])


class TestCombinedLossHK:
    def test_combined_loss_with_hk(self, small_cfg):
        """CombinedLoss includes H-K loss when epsilon_hk > 0."""
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        criterion = CombinedLoss(epsilon_hk=0.1, warmup_epochs=0)
        XYZ_1 = torch.randn(20, 3)
        XYZ_2 = torch.randn(20, 3)
        DV = torch.rand(20) + 0.1
        losses = criterion(model, XYZ_1, XYZ_2, DV)
        assert "hk" in losses
        assert losses["hk"].item() >= 0

    def test_combined_loss_hk_zero_weight(self, small_cfg):
        """When epsilon_hk=0, H-K loss should be 0."""
        small_cfg.inn_pad_dim = 0
        model = ColorINN(small_cfg)
        criterion = CombinedLoss(epsilon_hk=0.0, warmup_epochs=0)
        XYZ_1 = torch.randn(20, 3)
        XYZ_2 = torch.randn(20, 3)
        DV = torch.rand(20) + 0.1
        losses = criterion(model, XYZ_1, XYZ_2, DV)
        assert losses["hk"].item() == pytest.approx(0.0, abs=1e-10)
