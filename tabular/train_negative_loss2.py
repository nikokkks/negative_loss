import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_hardware(num_cpu_threads: int = 10) -> torch.device:
    """
    Configure PyTorch pour exploiter pleinement les 10 threads CPU et la VRAM/Tensor Cores du GPU.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_cpu_threads)
    torch.set_num_threads(num_cpu_threads)
    try:
        torch.set_num_interop_threads(min(num_cpu_threads, 4))
    except RuntimeError:
        pass

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible ! Vérifiez votre installation PyTorch/GPU."
        )

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return device


def set_seed(seed: int = 42) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPUTensorLoader:
    """
    DataLoader ultra-rapide résidant 100% en VRAM GPU.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        assert inputs.size(0) == targets.size(0)
        self.inputs = inputs.contiguous()
        self.targets = targets.contiguous()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = inputs.size(0)
        self.device = inputs.device

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            perm = torch.randperm(self.num_samples, device=self.device)
            inputs = self.inputs.index_select(0, perm)
            targets = self.targets.index_select(0, perm)
        else:
            inputs = self.inputs
            targets = self.targets

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            yield inputs[start:end], targets[start:end]


class TabularResBlock(nn.Module):
    """
    Bloc résiduel pour données tabulaires (inspiré de Tabular ResNet - Gorishniy et al., NeurIPS 2021) :
    x + Dropout(Linear(Dropout(GELU(Linear(LayerNorm(x))))))
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """
    Modèle Deep Learning pour données tabulaires (Tabular ResNet)
    adapté aux datasets tabulaires multi-classes (walking_22 / letter_26).
    """

    def __init__(
        self,
        num_features: int = 16,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        num_classes: int = 22,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def build_tabular_model(
    num_features: int = 16, num_classes: int = 22
) -> nn.Module:
    return TabularResNet(num_features=num_features, num_classes=num_classes)


def compute_macro_f1(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 26
) -> float:
    preds_oh = F.one_hot(preds.long(), num_classes=num_classes).to(torch.float32)
    targets_oh = F.one_hot(targets.long(), num_classes=num_classes).to(torch.float32)

    tp = (preds_oh * targets_oh).sum(dim=0)
    fp = (preds_oh * (1.0 - targets_oh)).sum(dim=0)
    fn = ((1.0 - preds_oh) * targets_oh).sum(dim=0)

    denom = 2.0 * tp + fp + fn
    f1_per_class = torch.where(
        denom > 0, (2.0 * tp) / denom.clamp(min=1e-12), torch.zeros_like(denom)
    )
    return float(f1_per_class.mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: GPUTensorLoader,
    num_classes: int = 26,
) -> Tuple[float, float]:
    model.eval()
    all_preds: List[torch.Tensor] = []

    for inputs, _ in val_loader:
        logits = model(inputs)
        all_preds.append(logits.argmax(dim=-1))

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = val_loader.targets

    macro_f1 = compute_macro_f1(preds_cat, targets_cat, num_classes=num_classes)
    accuracy = float((preds_cat == targets_cat).to(torch.float32).mean().item())
    return macro_f1, accuracy


def negative_unlikely_loss(
    logits: torch.Tensor,
    gt_targets: torch.Tensor,
    argmax_targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    batch_size, _ = logits.shape
    probs = F.softmax(logits, dim=-1)

    neg_mask = torch.ones_like(probs, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=logits.device)
    neg_mask[batch_indices, gt_targets] = False
    neg_mask[batch_indices, argmax_targets] = False

    log_one_minus_p = torch.log((1.0 - probs).clamp(min=eps))
    masked_log_probs = log_one_minus_p * neg_mask.to(probs.dtype)

    num_neg_per_sample = neg_mask.sum(dim=-1).clamp(min=1).to(probs.dtype)
    loss_per_sample = -masked_log_probs.sum(dim=-1) / num_neg_per_sample
    return loss_per_sample.mean()


def get_tabular_gpu_dataloaders(
    data_dir: Path,
    device: torch.device,
    dataset_name: str = "walking_22",
    num_train1: int = 1000,
    num_train2: int = 1000,
    num_val: int = 5000,
    batch_size: int = 64,
    val_batch_size: int = 1024,
    seed: int = 42,
) -> Tuple[GPUTensorLoader, GPUTensorLoader, GPUTensorLoader, int, int]:
    if dataset_name == "walking_22":
        pt_path = data_dir / "walking_22" / "walking22_tabular.pt"
    else:
        pt_path = data_dir / "letter_26" / "letter26_tabular.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Le dataset tabulaire {dataset_name} est introuvable dans {pt_path}."
        )

    payload = torch.load(pt_path, weights_only=True)
    train_features = payload["train_features"]
    train_targets = payload["train_targets"]
    val_features = payload["val_features"]
    val_targets = payload["val_targets"]
    num_features = int(payload["num_features"])
    num_classes = int(payload["num_classes"])

    rng = np.random.default_rng(seed)
    train_perm = torch.from_numpy(rng.permutation(len(train_features)))
    val_perm = torch.from_numpy(rng.permutation(len(val_features)))

    idx_phase1 = train_perm[:num_train1]
    idx_phase2 = train_perm[: num_train1 + num_train2]
    idx_val = val_perm[:num_val]

    p1_inputs = train_features.index_select(0, idx_phase1).to(device, non_blocking=True)
    p1_targets = train_targets.index_select(0, idx_phase1).to(device, non_blocking=True)

    p2_inputs = train_features.index_select(0, idx_phase2).to(device, non_blocking=True)
    p2_targets = train_targets.index_select(0, idx_phase2).to(device, non_blocking=True)

    v_inputs = val_features.index_select(0, idx_val).to(device, non_blocking=True)
    v_targets = val_targets.index_select(0, idx_val).to(device, non_blocking=True)

    loader_phase1 = GPUTensorLoader(
        p1_inputs, p1_targets, batch_size=batch_size, shuffle=True
    )
    loader_phase2 = GPUTensorLoader(
        p2_inputs, p2_targets, batch_size=batch_size, shuffle=True
    )
    loader_val = GPUTensorLoader(
        v_inputs, v_targets, batch_size=val_batch_size, shuffle=False
    )

    return loader_phase1, loader_phase2, loader_val, num_features, num_classes


def train_phase1_ce(
    model: nn.Module,
    train_loader: GPUTensorLoader,
    val_loader: GPUTensorLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_path: Path,
    num_classes: int = 26,
) -> Dict:
    print("=" * 80)
    print(
        f"PHASE 1 (Tabular - Letter-26, {num_classes} classes) : Entraînement Cross-Entropy sur 1000 lignes (VRAM Optimisé)"
    )
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
        total_samples = 0
        train_preds_list: List[torch.Tensor] = []
        train_targets_list: List[torch.Tensor] = []

        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss_gpu += loss.detach() * bs
            total_samples += bs
            train_preds_list.append(logits.detach().argmax(dim=-1))
            train_targets_list.append(targets)

        scheduler.step()
        avg_loss = float((running_loss_gpu / max(total_samples, 1)).item())
        train_preds_cat = torch.cat(train_preds_list, dim=0)
        train_targets_cat = torch.cat(train_targets_list, dim=0)
        train_f1 = compute_macro_f1(
            train_preds_cat, train_targets_cat, num_classes=num_classes
        )
        train_acc = float(
            (train_preds_cat == train_targets_cat).to(torch.float32).mean().item()
        )

        val_f1, val_acc = evaluate(
            model, val_loader, num_classes=num_classes
        )
        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "val_macro_f1": best_f1,
                    "val_acc": best_acc,
                },
                save_path,
            )

        marker = " * [NEW BEST]" if improved else ""
        print(
            f"[Phase 1 | Ep {epoch:03d}/{epochs:03d}] "
            f"Train(Loss={avg_loss:.4f}, F1={train_f1:.4f}, Acc={train_acc * 100:.2f}%) | "
            f"Val(F1={val_f1:.4f}, Acc={val_acc * 100:.2f}%) | "
            f"BestCkpt(Ep={best_epoch:03d}, F1={best_f1:.4f}, Acc={best_acc * 100:.2f}%, file={save_path.name})"
            f"{marker}"
        )

    model.load_state_dict(best_state)
    checkpoint_payload = {
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "val_macro_f1": best_f1,
        "val_acc": best_acc,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, save_path)
    print(
        f"\n-> Phase 1 terminée. Meilleur Checkpoint sauvegardé dans : {save_path} "
        f"(Ep={best_epoch:03d}, Val Macro-F1: {best_f1:.4f}, Val Acc: {best_acc * 100:.2f}%)\n"
    )
    return checkpoint_payload


def train_phase2_negative_loss(
    checkpoint_path: Path,
    train_loader: GPUTensorLoader,
    val_loader: GPUTensorLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience_epochs: int,
    ckpt_dir: Path,
    num_features: int = 16,
    num_classes: int = 26,
) -> None:
    """
    Phase 2 (Script 2 - SANS loss student) :
    Unique Loss : log(1 - p_negative) en excluant argmax du modèle référence et GT.
    """
    print("=" * 80)
    print(
        f"PHASE 2 (Tabular - Letter-26, {num_classes} classes) : Negative Learning REFERENCE ONLY (sans loss student, 2000 lignes en VRAM)"
    )
    print("=" * 80)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    init_state = base_ckpt["model_state_dict"]
    init_f1 = float(base_ckpt["val_macro_f1"])
    init_acc = float(base_ckpt.get("val_acc", 0.0))

    student_model = build_tabular_model(
        num_features=num_features, num_classes=num_classes
    ).to(device)
    ref_model = build_tabular_model(
        num_features=num_features, num_classes=num_classes
    ).to(device)

    student_model.load_state_dict(copy.deepcopy(init_state))
    ref_model.load_state_dict(copy.deepcopy(init_state))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ckpt_0_path = ckpt_dir / "ckpt_node_000_ep000.pt"
    init_state_copy = copy.deepcopy(init_state)
    torch.save(
        {
            "node_idx": 0,
            "epoch": 0,
            "val_macro_f1": init_f1,
            "val_acc": init_acc,
            "model_state_dict": init_state_copy,
        },
        ckpt_0_path,
    )
    checkpoint_tree: List[Dict] = [
        {
            "node_idx": 0,
            "epoch": 0,
            "val_macro_f1": init_f1,
            "val_acc": init_acc,
            "path": ckpt_0_path,
            "state_dict": init_state_copy,
        }
    ]
    current_tip = 0
    ref_node_idx = 0

    best_f1 = init_f1
    best_acc = init_acc
    best_epoch = 0
    best_node_idx = 0
    best_ckpt_name = ckpt_0_path.name
    epochs_without_improvement = 0

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
    )

    print(
        f"État initial Phase 2 chargé depuis {checkpoint_path} | "
        f"BestCkpt initial: Node 0 ({best_ckpt_name}) | Val Macro-F1: {best_f1:.4f} | Val Acc: {best_acc * 100:.2f}%"
    )

    for epoch in range(1, epochs + 1):
        student_model.train()
        ref_model.eval()

        running_loss_total = torch.zeros((), device=device, dtype=torch.float32)
        running_loss_ref = torch.zeros((), device=device, dtype=torch.float32)
        total_samples = 0
        train_preds_list: List[torch.Tensor] = []
        train_targets_list: List[torch.Tensor] = []

        for inputs, gt_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ref_logits = ref_model(inputs)
                ref_argmax = ref_logits.argmax(dim=-1)

            student_logits = student_model(inputs)
            with torch.no_grad():
                student_argmax = student_logits.argmax(dim=-1)

            loss_neg_ref = negative_unlikely_loss(
                student_logits, gt_targets, ref_argmax
            )

            loss = loss_neg_ref
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss_total += loss.detach() * bs
            running_loss_ref += loss_neg_ref.detach() * bs
            total_samples += bs
            train_preds_list.append(student_argmax)
            train_targets_list.append(gt_targets)

        inv_total = 1.0 / max(total_samples, 1)
        avg_loss_total = float((running_loss_total * inv_total).item())
        avg_loss_ref = float((running_loss_ref * inv_total).item())

        train_preds_cat = torch.cat(train_preds_list, dim=0)
        train_targets_cat = torch.cat(train_targets_list, dim=0)
        train_f1 = compute_macro_f1(
            train_preds_cat, train_targets_cat, num_classes=num_classes
        )
        train_acc = float(
            (train_preds_cat == train_targets_cat).to(torch.float32).mean().item()
        )

        val_f1, val_acc = evaluate(
            student_model, val_loader, num_classes=num_classes
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            if current_tip < len(checkpoint_tree) - 1:
                checkpoint_tree = checkpoint_tree[: current_tip + 1]

            new_node_idx = len(checkpoint_tree)
            new_ckpt_path = ckpt_dir / f"ckpt_node_{new_node_idx:03d}_ep{epoch:03d}.pt"
            new_state = copy.deepcopy(student_model.state_dict())
            torch.save(
                {
                    "node_idx": new_node_idx,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                    "model_state_dict": new_state,
                },
                new_ckpt_path,
            )
            checkpoint_tree.append(
                {
                    "node_idx": new_node_idx,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                    "path": new_ckpt_path,
                    "state_dict": new_state,
                }
            )
            current_tip = new_node_idx
            ref_node_idx = new_node_idx
            best_node_idx = new_node_idx
            best_ckpt_name = new_ckpt_path.name

            ref_model.load_state_dict(copy.deepcopy(new_state))
            ref_model.eval()

            best_phase2_path = ckpt_dir / "best_phase2_model.pt"
            torch.save(
                {
                    "node_idx": new_node_idx,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                    "model_state_dict": new_state,
                },
                best_phase2_path,
            )

            print(
                f"[Phase 2 | Ep {epoch:03d}/{epochs:03d}] "
                f"Train(Loss={avg_loss_total:.4f} [Ref={avg_loss_ref:.4f}], F1={train_f1:.4f}, Acc={train_acc * 100:.2f}%) | "
                f"Val(F1={val_f1:.4f}, Acc={val_acc * 100:.2f}%) | "
                f"BestCkpt(Node={best_node_idx}, Ep={best_epoch:03d}, F1={best_f1:.4f}, Acc={best_acc * 100:.2f}%, file={best_ckpt_name}) | "
                f"ActiveRef=Node {ref_node_idx} * [NEW BEST]"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[Phase 2 | Ep {epoch:03d}/{epochs:03d}] "
                f"Train(Loss={avg_loss_total:.4f} [Ref={avg_loss_ref:.4f}], F1={train_f1:.4f}, Acc={train_acc * 100:.2f}%) | "
                f"Val(F1={val_f1:.4f}, Acc={val_acc * 100:.2f}%) | "
                f"BestCkpt(Node={best_node_idx}, Ep={best_epoch:03d}, F1={best_f1:.4f}, Acc={best_acc * 100:.2f}%, file={best_ckpt_name}) | "
                f"ActiveRef=Node {ref_node_idx} (Tip t={current_tip}) | "
                f"NoImprov={epochs_without_improvement}/{patience_epochs}"
            )

            if (
                epochs_without_improvement > 0
                and epochs_without_improvement % patience_epochs == 0
            ):
                student_target_idx = max(0, current_tip - 1)
                ref_target_idx = max(0, current_tip - 2)

                student_node = checkpoint_tree[student_target_idx]
                ref_node = checkpoint_tree[ref_target_idx]

                student_model.load_state_dict(
                    copy.deepcopy(student_node["state_dict"])
                )
                ref_model.load_state_dict(
                    copy.deepcopy(ref_node["state_dict"])
                )
                ref_model.eval()

                prev_tip = current_tip
                current_tip = student_target_idx
                ref_node_idx = ref_target_idx

                print(
                    f"  >>> [ROLLBACK ARBORESCENCE @ {epochs_without_improvement} époques sans amélioration] "
                    f"Pointeur t={prev_tip} -> Student chargé depuis t-1 (Node {student_target_idx}, "
                    f"ep={student_node['epoch']}, F1={student_node['val_macro_f1']:.4f}, file={student_node['path'].name}) | "
                    f"Référence chargée depuis t-2 (Node {ref_target_idx}, "
                    f"ep={ref_node['epoch']}, F1={ref_node['val_macro_f1']:.4f}, file={ref_node['path'].name}). "
                    f"Nouveau pointeur t={current_tip}."
                )

    print("\n" + "=" * 80)
    print(
        f"PHASE 2 TERMINÉE. Meilleur Val Macro-F1 atteint : {best_f1:.4f} "
        f"(Nombre total de noeuds dans l'arborescence : {len(checkpoint_tree)})"
    )
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement Tabular Letter-26 (26 classes) - Script 2 (SANS loss student, uniquement loss référence)."
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["both", "phase1", "phase2"],
        default="both",
        help="Exécuter les deux phases ('both'), uniquement la Phase 1 ('phase1'), ou uniquement la Phase 2 ('phase2').",
    )
    parser.add_argument(
        "--num-train1",
        type=int,
        default=1000,
        help="Nombre de données pour le train 1 (Phase 1 Cross-Entropy).",
    )
    parser.add_argument(
        "--num-train2",
        type=int,
        default=1000,
        help="Nombre de données supplémentaires ajoutées au train 1 pour le train 2 (Phase 2 Negative Loss : total = num_train1 + num_train2).",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=5000,
        help="Nombre de données de validation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["walking_22", "letter_26"],
        default="walking_22",
        help="Dataset tabulaire à utiliser : walking_22 (22 classes, difficile ~37% F1) ou letter_26 (26 classes, ~84% F1).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Répertoire contenant les datasets tabulaires.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints_letter26",
        help="Répertoire de sauvegarde des checkpoints.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=50,
        help="Nombre d'époques pour la Phase 1 (Cross-Entropy sur 1000 lignes).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=120,
        help="Nombre d'époques pour la Phase 2 (Negative Loss Reference Only sur 2000 lignes).",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=400,
        help="Nombre d'époques sans amélioration avant de remonter l'arborescence (t-1 pour student, t-2 pour référence).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille de batch d'entraînement.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1024,
        help="Taille de batch de validation en VRAM.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=10,
        help="Nombre de threads CPU utilisés par PyTorch.",
    )
    parser.add_argument(
        "--lr-phase1",
        type=float,
        default=1e-3,
        help="Learning rate pour la Phase 1 (CE).",
    )
    parser.add_argument(
        "--lr-phase2",
        type=float,
        default=1e-4,
        help="Learning rate pour la Phase 2 (Negative Loss).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay pour AdamW.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = configure_hardware(num_cpu_threads=args.cpu_threads)
    set_seed(args.seed)

    print(
        f"Utilisation du GPU : {torch.cuda.get_device_name(device)} ({device}) | "
        f"Threads CPU : {torch.get_num_threads()} | TF32 + Fused AdamW + 100% VRAM Tabular Dataset"
    )

    _ = torch.zeros(1, device=device)

    loader_phase1, loader_phase2, loader_val, num_features, num_classes = (
        get_tabular_gpu_dataloaders(
            data_dir=args.data_dir,
            device=device,
            dataset_name=args.dataset,
            num_train1=args.num_train1,
            num_train2=args.num_train2,
            num_val=args.num_val,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            seed=args.seed,
        )
    )

    phase1_ckpt_path = args.ckpt_dir / "phase1_ce_1000.pt"

    if args.phase in ("both", "phase1"):
        model_phase1 = build_tabular_model(
            num_features=num_features, num_classes=num_classes
        ).to(device)
        train_phase1_ce(
            model=model_phase1,
            train_loader=loader_phase1,
            val_loader=loader_val,
            device=device,
            epochs=args.phase1_epochs,
            lr=args.lr_phase1,
            weight_decay=args.weight_decay,
            save_path=phase1_ckpt_path,
            num_classes=num_classes,
        )

    if args.phase in ("both", "phase2"):
        train_phase2_negative_loss(
            checkpoint_path=phase1_ckpt_path,
            train_loader=loader_phase2,
            val_loader=loader_val,
            device=device,
            epochs=args.phase2_epochs,
            lr=args.lr_phase2,
            weight_decay=args.weight_decay,
            patience_epochs=args.patience_epochs,
            ckpt_dir=args.ckpt_dir / "phase2_tree_loss2",
            num_features=num_features,
            num_classes=num_classes,
        )


if __name__ == "__main__":
    main()
