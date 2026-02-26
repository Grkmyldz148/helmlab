"""Loss functions for neural color space training."""

import torch
import torch.nn as nn


class STRESSLoss(nn.Module):
    """Differentiable STRESS loss.

    STRESS = 100 * sqrt( sum((DV - F*DE)^2) / sum(DV^2) )
    where F = sum(DV*DE) / sum(DE^2) is the optimal scaling factor.
    """

    def forward(
        self,
        perceptual_1: torch.Tensor,
        perceptual_2: torch.Tensor,
        DV: torch.Tensor,
    ) -> torch.Tensor:
        DE = torch.sqrt(torch.sum((perceptual_1 - perceptual_2) ** 2, dim=-1) + 1e-12)
        DE_sq_sum = torch.sum(DE ** 2) + 1e-12
        F = (torch.sum(DV * DE) / DE_sq_sum).detach()
        residual = DV - F * DE
        DV_sq_sum = torch.sum(DV ** 2) + 1e-12
        return 100.0 * torch.sqrt(torch.sum(residual ** 2) / DV_sq_sum)


class RankingLoss(nn.Module):
    """Soft ranking loss: encourage DE to respect the ordering of DV.

    For each pair of pairs (i, j) in the batch where DV_i > DV_j,
    we want DE_i > DE_j. Uses a soft margin loss.
    """

    def __init__(self, n_samples: int = 512, margin: float = 0.01):
        super().__init__()
        self.n_samples = n_samples
        self.margin = margin

    def forward(
        self,
        perceptual_1: torch.Tensor,
        perceptual_2: torch.Tensor,
        DV: torch.Tensor,
    ) -> torch.Tensor:
        DE = torch.sqrt(torch.sum((perceptual_1 - perceptual_2) ** 2, dim=-1) + 1e-12)

        B = DE.shape[0]
        n = min(self.n_samples, B * (B - 1) // 2)

        # Random pair indices
        idx_i = torch.randint(0, B, (n,), device=DE.device)
        idx_j = torch.randint(0, B, (n,), device=DE.device)

        dv_diff = DV[idx_i] - DV[idx_j]   # positive means DV_i > DV_j
        de_diff = DE[idx_i] - DE[idx_j]

        # Where DV_i > DV_j, DE_i should also be > DE_j (and vice versa)
        # Loss: max(0, margin - sign(dv_diff) * de_diff)
        sign = torch.sign(dv_diff)
        # Filter out near-zero dv_diff (same DV → no ranking constraint)
        mask = dv_diff.abs() > 1e-6
        if mask.sum() == 0:
            return torch.tensor(0.0, device=DE.device, dtype=DE.dtype)

        violations = torch.clamp(self.margin - sign[mask] * de_diff[mask], min=0.0)
        return violations.mean()


class LogScaleLoss(nn.Module):
    """MSE between log-scaled predicted and target distances.

    More numerically stable for initial training than STRESS.
    """

    def forward(
        self,
        perceptual_1: torch.Tensor,
        perceptual_2: torch.Tensor,
        DV: torch.Tensor,
    ) -> torch.Tensor:
        DE = torch.sqrt(torch.sum((perceptual_1 - perceptual_2) ** 2, dim=-1) + 1e-12)
        log_DE = torch.log(DE + 1e-8)
        log_DV = torch.log(DV + 1e-8)
        # Normalize: remove mean (makes it scale-invariant)
        log_DE_centered = log_DE - log_DE.mean()
        log_DV_centered = log_DV - log_DV.mean()
        return torch.mean((log_DE_centered - log_DV_centered) ** 2)


class HueLinearityLoss(nn.Module):
    """Loss for hue linearity: constant-hue points should be collinear in a-b plane."""

    def forward(
        self,
        perceptual: torch.Tensor,
        hue_idx: torch.Tensor,
        n_hues: int = 12,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=perceptual.device, dtype=perceptual.dtype)
        count = 0

        for h in range(n_hues):
            mask = hue_idx == h
            if mask.sum() < 3:
                continue

            ab = perceptual[mask, 1:3]
            ab_centered = ab - ab.mean(dim=0, keepdim=True)

            cov = ab_centered.T @ ab_centered
            tr = cov[0, 0] + cov[1, 1]
            det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
            disc = torch.clamp(tr * tr - 4 * det, min=1e-12)
            lambda_min = (tr - torch.sqrt(disc)) / 2.0
            total_loss = total_loss + lambda_min / (tr + 1e-12)
            count += 1

        if count == 0:
            return total_loss
        return total_loss / count


class RoundTripLoss(nn.Module):
    """Round-trip reconstruction loss for MLP."""

    def forward(self, XYZ_original: torch.Tensor, XYZ_reconstructed: torch.Tensor) -> torch.Tensor:
        return torch.mean((XYZ_original - XYZ_reconstructed) ** 2)


class HKEffectLoss(nn.Module):
    """Helmholtz-Kohlrausch effect loss.

    Enforces: at the same luminance Y, chromatic colors should have higher
    perceived lightness (L) than achromatic colors.

    Generates on-the-fly achromatic/chromatic pairs and applies a margin loss:
        loss = max(0, margin - (L_chromatic - L_achromatic))

    The H-K effect is strongest for blue and red, weaker for yellow.
    """

    def __init__(self, margin: float = 0.05, n_samples: int = 64):
        super().__init__()
        self.margin = margin
        self.n_samples = n_samples
        # D65 white chromaticity
        self.x_d65 = 0.3127
        self.y_d65 = 0.3290

    def _generate_hk_pairs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate achromatic/chromatic XYZ pairs at same Y."""
        n = self.n_samples
        Y = torch.rand(n, device=device) * 0.8 + 0.1  # Y in [0.1, 0.9]
        hue_angle = torch.rand(n, device=device) * 2 * 3.14159
        chroma_scale = torch.rand(n, device=device) * 0.12 + 0.03  # [0.03, 0.15]

        # Achromatic: D65 chromaticity
        x_a = self.x_d65
        y_a = self.y_d65
        X_a = x_a * Y / y_a
        Z_a = (1 - x_a - y_a) * Y / y_a
        XYZ_achromatic = torch.stack([X_a, Y, Z_a], dim=-1)

        # Chromatic: offset from D65
        x_c = self.x_d65 + chroma_scale * torch.cos(hue_angle)
        y_c = torch.clamp(self.y_d65 + chroma_scale * torch.sin(hue_angle), min=0.01)
        X_c = x_c * Y / y_c
        Z_c = (1 - x_c - y_c) * Y / y_c
        XYZ_chromatic = torch.stack([X_c, Y, Z_c], dim=-1)

        return XYZ_achromatic, XYZ_chromatic

    def forward(self, model) -> torch.Tensor:
        device = next(model.parameters()).device
        XYZ_a, XYZ_c = self._generate_hk_pairs(device)

        L_a = model(XYZ_a)[:, 0]  # achromatic lightness
        L_c = model(XYZ_c)[:, 0]  # chromatic lightness

        # Chromatic should have higher L than achromatic (H-K effect)
        violations = torch.clamp(self.margin - (L_c - L_a), min=0.0)
        return violations.mean()


class D4RegularizationLoss(nn.Module):
    """Regularize INN 4th dimension output toward zero.

    Uses mean(d4^2) + max(d4^2) to penalize both average and outlier d4.
    When d4 ≈ 0, the approximate inverse (padding d4=0) becomes accurate,
    fixing the catastrophic round-trip error of unregularized INNs.
    """

    def forward(self, d4: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(d4 ** 2)
        max_sq = torch.max(d4 ** 2)
        return mean_sq + max_sq


class CombinedLoss(nn.Module):
    """Combined loss with two-phase training support.

    Phase 1 (warmup): ranking + log-scale loss (easy to optimize, learns ordering)
    Phase 2 (refine): STRESS loss (precise metric alignment)

    For INN models with forward_full(), also applies d4 regularization to
    push the 4th latent dimension toward zero for accurate round-trip.
    """

    def __init__(
        self,
        alpha_stress: float = 1.0,
        alpha_ranking: float = 1.0,
        alpha_logscale: float = 1.0,
        beta_hue: float = 0.1,
        gamma_roundtrip: float = 0.0,
        delta_d4: float = 1.0,
        epsilon_hk: float = 0.0,
        warmup_epochs: int = 100,
    ):
        super().__init__()
        self.alpha_stress = alpha_stress
        self.alpha_ranking = alpha_ranking
        self.alpha_logscale = alpha_logscale
        self.beta_hue = beta_hue
        self.gamma_roundtrip = gamma_roundtrip
        self.delta_d4 = delta_d4
        self.epsilon_hk = epsilon_hk
        self.warmup_epochs = warmup_epochs

        self.stress_loss = STRESSLoss()
        self.ranking_loss = RankingLoss()
        self.logscale_loss = LogScaleLoss()
        self.hue_loss = HueLinearityLoss()
        self.round_trip_loss = RoundTripLoss()
        self.d4_reg_loss = D4RegularizationLoss()
        self.hk_loss = HKEffectLoss()

        self.current_epoch = 0

    def _get_perceptual_and_d4(
        self, model, XYZ: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get perceptual coords and optional d4 from model.

        For 4D INN models (pad_dim>0, with forward_full), returns 3D perceptual and 1D d4.
        For 3D models or other models, returns 3D perceptual and None.
        """
        if (self.delta_d4 > 0
                and hasattr(model, "forward_full")
                and getattr(model, "pad_dim", 0) > 0):
            z_full, _ = model.forward_full(XYZ)
            return z_full[:, :3], z_full[:, 3]
        return model(XYZ), None

    def forward(
        self,
        model,
        XYZ_1: torch.Tensor,
        XYZ_2: torch.Tensor,
        DV: torch.Tensor,
        hue_XYZ: torch.Tensor | None = None,
        hue_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        p1, d4_1 = self._get_perceptual_and_d4(model, XYZ_1)
        p2, d4_2 = self._get_perceptual_and_d4(model, XYZ_2)

        losses = {}
        total = torch.tensor(0.0, device=XYZ_1.device, dtype=p1.dtype)

        # Phase blending: warmup uses ranking+logscale, then transitions to STRESS
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            w_ranking = 1.0 - progress   # fades from 1→0
            w_stress = progress           # fades from 0→1

            loss_ranking = self.ranking_loss(p1, p2, DV)
            loss_logscale = self.logscale_loss(p1, p2, DV)
            loss_stress = self.stress_loss(p1, p2, DV)

            total = (w_ranking * (self.alpha_ranking * loss_ranking + self.alpha_logscale * loss_logscale)
                     + w_stress * self.alpha_stress * loss_stress)

            losses["ranking"] = loss_ranking.detach()
            losses["logscale"] = loss_logscale.detach()
            losses["stress"] = loss_stress.detach()
        else:
            loss_stress = self.stress_loss(p1, p2, DV)
            total = self.alpha_stress * loss_stress
            losses["stress"] = loss_stress.detach()

        # Hue linearity
        loss_hue = torch.tensor(0.0, device=XYZ_1.device)
        if hue_XYZ is not None and hue_idx is not None and self.beta_hue > 0:
            p_hue = model(hue_XYZ)
            loss_hue = self.hue_loss(p_hue, hue_idx)
            total = total + self.beta_hue * loss_hue
        losses["hue"] = loss_hue.detach()

        # d4 regularization (INN only) — ramps up over training
        loss_d4 = torch.tensor(0.0, device=XYZ_1.device)
        if d4_1 is not None and d4_2 is not None and self.delta_d4 > 0:
            d4_all = torch.cat([d4_1, d4_2])
            loss_d4 = self.d4_reg_loss(d4_all)
            # Ramp: start at 0.1x during warmup, reach full weight after warmup
            if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
                d4_scale = 0.1 + 0.9 * (self.current_epoch / self.warmup_epochs)
            else:
                d4_scale = 1.0
            total = total + self.delta_d4 * d4_scale * loss_d4
        losses["d4_reg"] = loss_d4.detach()

        # H-K effect loss (chromatic colors should have higher L)
        loss_hk = torch.tensor(0.0, device=XYZ_1.device)
        if self.epsilon_hk > 0:
            loss_hk = self.hk_loss(model)
            total = total + self.epsilon_hk * loss_hk
        losses["hk"] = loss_hk.detach()

        # Round-trip loss (MLP via decoder, INN via forward→inverse 3D path)
        loss_rt = torch.tensor(0.0, device=XYZ_1.device)
        if self.gamma_roundtrip > 0:
            XYZ_cat = torch.cat([XYZ_1, XYZ_2], dim=0)
            p_cat = model(XYZ_cat)
            XYZ_rec = model.inverse(p_cat)
            loss_rt = self.round_trip_loss(XYZ_cat, XYZ_rec)
            total = total + self.gamma_roundtrip * loss_rt
        losses["roundtrip"] = loss_rt.detach()

        losses["total"] = total
        return losses
