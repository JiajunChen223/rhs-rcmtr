import torch


def test_candidate_uses_six_execution_epochs_but_twelve_epoch_scheduler() -> None:
    parameter_a = torch.nn.Parameter(torch.ones(()))
    parameter_b = torch.nn.Parameter(torch.ones(()))
    optimizer_a = torch.optim.AdamW([parameter_a], lr=1e-4, weight_decay=0.01)
    optimizer_b = torch.optim.AdamW([parameter_b], lr=1e-4, weight_decay=0.01)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=12)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=12)
    lrs_a, lrs_b = [], []
    for _ in range(6):
        lrs_a.append(optimizer_a.param_groups[0]["lr"])
        lrs_b.append(optimizer_b.param_groups[0]["lr"])
        scheduler_a.step()
        scheduler_b.step()
    assert lrs_a == lrs_b
    assert lrs_a[-1] > 0
    assert scheduler_a.last_epoch == 6


def test_explicit_loss_objective_is_configurable() -> None:
    from rhs_rcmtr.losses import segmentation_with_honesty_loss

    logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    route = torch.full((1, 2, 4, 4), 0.5)
    reliability = torch.zeros(1, 2, 4, 4)
    reliability[:, 0] = 0.9
    reliability[:, 1] = 0.1
    plain, _ = segmentation_with_honesty_loss(
        logits, labels, route, reliability, honesty_weight=0.0, segmentation_weight=1.0
    )
    honest, _ = segmentation_with_honesty_loss(
        logits, labels, route, reliability, honesty_weight=0.1, segmentation_weight=1.0
    )
    assert honest > plain


def test_diagnostic_run_may_continue_past_rapid_checkpoint() -> None:
    import yaml
    from pathlib import Path

    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiment"
            / "n1_d2_sar_only.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["diagnostic_only"] is True
    assert config["execution_epochs"] == 12
    assert config["rapid_screen_horizon_epochs"] == 6


def test_oa_scrt_formal_config_cannot_silently_reuse_rapid_horizon() -> None:
    import yaml
    from pathlib import Path

    config_root = Path(__file__).resolve().parents[2] / "configs"
    rapid = yaml.safe_load(
        (config_root / "experiment" / "r5_c1_oa_scrt.yaml").read_text(encoding="utf-8")
    )
    formal = yaml.safe_load(
        (config_root / "experiment" / "r5_c1_oa_scrt_formal.yaml").read_text(encoding="utf-8")
    )
    runtime = yaml.safe_load(
        (config_root / "runtime" / "cloud_formal_screening.yaml").read_text(encoding="utf-8")
    )

    assert rapid["execution_epochs"] == rapid["rapid_screen_horizon_epochs"] == 6
    assert formal["epochs"] == formal["execution_epochs"] == formal["scheduler_horizon_epochs"] == 12
    assert formal["rapid_screen_horizon_epochs"] == 6
    assert runtime["execution_scale"] == "innovation_formal_seed0"
    assert runtime["test_seal_status"] == "sealed"
    assert runtime["evaluation_splits"] == ["train", "validation"]
