"""Training loop for neural color space models."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from helmlab.config import TrainConfig, CHECKPOINT_DIR
from helmlab.nn.inn import ColorINN
from helmlab.nn.mlp import ColorMLP
from helmlab.nn.losses import CombinedLoss
from helmlab.data.dataset import build_dataloaders


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_type: str, cfg: TrainConfig) -> nn.Module:
    if model_type == "inn":
        return ColorINN(cfg)
    elif model_type == "mlp":
        return ColorMLP(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(
    model_type: str = "inn",
    cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> dict:
    if cfg is None:
        cfg = TrainConfig()

    console = Console()
    device = get_device()
    if verbose:
        console.print(f"[bold]Device:[/bold] {device}")

    torch.manual_seed(cfg.seed)

    # Data
    if verbose:
        console.print("[bold]Loading data...[/bold]")
    loaders = build_dataloaders(cfg)
    train_loader = loaders["train_pairs"]
    val_loader = loaders["val_pairs"]
    hue_loader = loaders["hue"]

    hue_batch = next(iter(hue_loader))
    hue_XYZ = hue_batch[0].to(device)
    hue_idx = hue_batch[1].to(device)

    if verbose:
        console.print(f"[bold]Train pairs:[/bold] {loaders['n_train']}, "
                      f"[bold]Val pairs:[/bold] {loaders['n_val']}")

    # Model
    model = build_model(model_type, cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        console.print(f"[bold]Model:[/bold] {model_type.upper()}, params: {n_params:,}")

    # Loss — two-phase: warmup with ranking+logscale, then pure STRESS
    warmup_epochs = cfg.epochs // 3  # first third is warmup
    gamma = cfg.gamma_roundtrip
    delta_d4 = cfg.delta_d4 if (model_type == "inn" and cfg.inn_pad_dim > 0) else 0.0
    epsilon_hk = cfg.epsilon_hk if hasattr(cfg, "epsilon_hk") else 0.0
    criterion = CombinedLoss(
        alpha_stress=cfg.alpha_stress,
        alpha_ranking=1.0,
        alpha_logscale=1.0,
        beta_hue=cfg.beta_hue,
        gamma_roundtrip=gamma,
        delta_d4=delta_d4,
        epsilon_hk=epsilon_hk,
        warmup_epochs=warmup_epochs,
    )

    if verbose:
        console.print(f"[bold]Warmup:[/bold] {warmup_epochs} epochs (ranking+logscale → STRESS)")
        if delta_d4 > 0:
            console.print(f"[bold]d4 regularization:[/bold] weight={delta_d4}")
        if epsilon_hk > 0:
            console.print(f"[bold]H-K effect loss:[/bold] weight={epsilon_hk}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult)

    # Training
    history = {"train_loss": [], "val_stress": [], "lr": []}
    best_val_stress = float("inf")
    best_val_score = float("inf")  # STRESS + d4 penalty (for checkpoint selection)
    best_epoch = 0
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dim_suffix = f"_{model.total_dim}d" if hasattr(model, "total_dim") else ""
    best_path = CHECKPOINT_DIR / f"{model_type}{dim_suffix}_best.pt"

    progress_cols = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
    ]

    with Progress(*progress_cols, console=console, disable=not verbose) as progress:
        task = progress.add_task("Training", total=cfg.epochs, info="")

        for epoch in range(1, cfg.epochs + 1):
            criterion.current_epoch = epoch - 1

            # ── Train ────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for XYZ_1, XYZ_2, DV in train_loader:
                XYZ_1 = XYZ_1.to(device)
                XYZ_2 = XYZ_2.to(device)
                DV = DV.to(device)

                optimizer.zero_grad()
                losses = criterion(model, XYZ_1, XYZ_2, DV, hue_XYZ, hue_idx)
                loss = losses["total"]
                loss.backward()

                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # ── Validate ─────────────────────────────────
            model.eval()
            all_DV = []
            all_DE = []
            all_d4 = []
            use_d4 = delta_d4 > 0 and hasattr(model, "forward_full") and getattr(model, "pad_dim", 0) > 0

            with torch.no_grad():
                for XYZ_1, XYZ_2, DV in val_loader:
                    XYZ_1 = XYZ_1.to(device)
                    XYZ_2 = XYZ_2.to(device)
                    if use_d4:
                        z1_full, _ = model.forward_full(XYZ_1)
                        z2_full, _ = model.forward_full(XYZ_2)
                        p1 = z1_full[:, :3]
                        p2 = z2_full[:, :3]
                        all_d4.append(z1_full[:, 3].cpu())
                        all_d4.append(z2_full[:, 3].cpu())
                    else:
                        p1 = model(XYZ_1)
                        p2 = model(XYZ_2)
                    DE = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1) + 1e-12)
                    all_DV.append(DV)
                    all_DE.append(DE.cpu())

            all_DV = torch.cat(all_DV)
            all_DE = torch.cat(all_DE)

            DE_sq = torch.sum(all_DE ** 2) + 1e-12
            F = torch.sum(all_DV * all_DE) / DE_sq
            residual = all_DV - F * all_DE
            val_stress = 100.0 * torch.sqrt(torch.sum(residual ** 2) / (torch.sum(all_DV ** 2) + 1e-12))
            val_stress = val_stress.item()
            history["val_stress"].append(val_stress)

            # Compute d4-penalized validation score for checkpoint selection
            val_d4_mean = 0.0
            if use_d4 and all_d4:
                all_d4_t = torch.cat(all_d4)
                val_d4_mean = torch.mean(all_d4_t ** 2).item()
            # Combined metric: STRESS + weighted d4 penalty
            val_score = val_stress + delta_d4 * val_d4_mean

            # Early stopping (only after warmup)
            if val_score < best_val_score:
                best_val_score = val_score
                best_val_stress = val_stress
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_stress": val_stress,
                    "val_d4_mse": val_d4_mean,
                    "config": cfg.to_dict(),
                    "model_type": model_type,
                }, best_path)
            else:
                if epoch > warmup_epochs:
                    patience_counter += 1

            phase = "warmup" if epoch <= warmup_epochs else "STRESS"
            d4_info = f" d4={val_d4_mean:.4f}" if use_d4 else ""
            info = f"[{phase}] loss={avg_train_loss:.3f} val_STRESS={val_stress:.2f}{d4_info} best={best_val_stress:.2f}"
            progress.update(task, advance=1, info=info)

            # Log every 25 epochs for nohup visibility
            if epoch % 25 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{cfg.epochs} {info}", flush=True)

            if patience_counter >= cfg.early_stop_patience:
                if verbose:
                    console.print(f"\n[yellow]Early stopping at epoch {epoch}[/yellow]")
                break

    # Load best model
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if verbose:
        console.print(f"\n[green]Best epoch: {best_epoch}, Val STRESS: {best_val_stress:.2f}[/green]")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_stress": best_val_stress,
        "device": device,
    }
