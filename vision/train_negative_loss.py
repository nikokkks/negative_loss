import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def set_seed(seed: int = 42) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LightResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class LightResNet9Cifar10(nn.Module):
    """
    ResNet-9 léger adapté aux images 32x32 de CIFAR-10 (~0.70M paramètres contre 11.17M pour ResNet-18,
    soit 16x plus léger et ~5x plus rapide en entraînement/inférence).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = LightResBlock(32, 64, stride=2)    # 16x16
        self.layer2 = LightResBlock(64, 128, stride=2)   # 8x8
        self.layer3 = LightResBlock(128, 128, stride=2)  # 4x4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    Construit le modèle vision léger LightResNet9Cifar10 (~0.70M paramètres) pour CIFAR-10.
    """
    return LightResNet9Cifar10(num_classes=num_classes)


def compute_macro_f1(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 10
) -> float:
    """
    Calcule le F1-score Macro de manière 100% vectorisée sur GPU (CUDA).
    """
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
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[float, float]:
    """
    Évalue le modèle sur les 5000 données de validation directement sur GPU.
    Retourne (macro_f1, accuracy).
    """
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for inputs, targets in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds)
        all_targets.append(targets)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    macro_f1 = compute_macro_f1(preds_cat, targets_cat, num_classes=num_classes)
    accuracy = float((preds_cat == targets_cat).to(torch.float32).mean().item())
    return macro_f1, accuracy


def negative_unlikely_loss(
    logits: torch.Tensor,
    gt_targets: torch.Tensor,
    argmax_targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calcule la loss négative en excluant la classe GT et la classe argmax donnée :
        L = - mean_{c not in {GT, argmax}} log(1 - p_c)
    où p = softmax(logits).
    """
    batch_size, _ = logits.shape
    probs = F.softmax(logits, dim=-1)

    # Masque des classes négatives : True partout sauf sur GT et sur argmax_targets
    neg_mask = torch.ones_like(probs, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=logits.device)
    neg_mask[batch_indices, gt_targets] = False
    neg_mask[batch_indices, argmax_targets] = False

    # log(1 - p_c) uniquement sur les classes négatives
    log_one_minus_p = torch.log((1.0 - probs).clamp(min=eps))
    masked_log_probs = log_one_minus_p * neg_mask.to(probs.dtype)

    # Moyenne par échantillon sur le nombre de classes négatives actives (8 ou 9 sur CIFAR-10), puis moyenne batch
    num_neg_per_sample = neg_mask.sum(dim=-1).clamp(min=1).to(probs.dtype)
    loss_per_sample = -masked_log_probs.sum(dim=-1) / num_neg_per_sample
    return loss_per_sample.mean()


def get_cifar10_dataloaders(
    data_dir: Path,
    num_train1: int = 1000,
    num_train2: int = 1000,
    num_val: int = 5000,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prépare les datasets CIFAR-10 :
    - Phase 1 : `num_train1` premières données d'entraînement (ex: 1000)
    - Phase 2 : les `num_train1` données de la Phase 1 + `num_train2` données supplémentaires (ex: 1000 + 1000 = 2000)
    - Validation : `num_val` données de validation (ex: 5000)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    extracted_dir = data_dir / "cifar-10-batches-py"
    if (extracted_dir / "data_batch_1").exists() and (extracted_dir / "test_batch").exists():
        # Court-circuite la vérification MD5 de torchvision si les fichiers extraits sont déjà présents
        torchvision.datasets.CIFAR10._check_integrity = lambda self: True
        torchvision.datasets.cifar.check_integrity = lambda *args, **kwargs: True
        should_download = False
    else:
        should_download = True

    full_train = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=should_download,
        transform=train_transform,
    )
    full_test = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=should_download,
        transform=eval_transform,
    )

    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(full_train)).tolist()
    val_perm = rng.permutation(len(full_test)).tolist()

    idx_phase1 = train_perm[:num_train1]
    idx_phase2 = train_perm[: num_train1 + num_train2]
    idx_val = val_perm[:num_val]

    phase1_subset = Subset(full_train, idx_phase1)
    phase2_subset = Subset(full_train, idx_phase2)
    val_subset = Subset(full_test, idx_val)

    loader_phase1 = DataLoader(
        phase1_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    loader_phase2 = DataLoader(
        phase2_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    loader_val = DataLoader(
        val_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader_phase1, loader_phase2, loader_val


def train_phase1_ce(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    save_path: Path,
) -> Dict:
    """
    Phase 1 : Entraînement par Cross-Entropy classique sur 1000 données CIFAR-10,
    puis sauvegarde du meilleur checkpoint.
    """
    print("=" * 80)
    print("PHASE 1 : Entraînement Cross-Entropy sur 1000 données CIFAR-10")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
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
        running_loss = 0.0
        total_samples = 0
        train_preds_list: List[torch.Tensor] = []
        train_targets_list: List[torch.Tensor] = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            train_preds_list.append(logits.detach().argmax(dim=-1))
            train_targets_list.append(targets.detach())

        scheduler.step()
        avg_loss = running_loss / max(total_samples, 1)
        train_preds_cat = torch.cat(train_preds_list, dim=0)
        train_targets_cat = torch.cat(train_targets_list, dim=0)
        train_f1 = compute_macro_f1(train_preds_cat, train_targets_cat)
        train_acc = float(
            (train_preds_cat == train_targets_cat).to(torch.float32).mean().item()
        )

        val_f1, val_acc = evaluate(model, val_loader, device)
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

    # On restaure et sauvegarde le meilleur état de la Phase 1
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
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience_epochs: int,
    ckpt_dir: Path,
) -> None:
    """
    Phase 2 : Entraînement sur 2000 données CIFAR-10 (les 1000 de la Phase 1 + 1000 nouvelles)
    en chargeant le checkpoint précédent, SANS Cross-Entropy loss :
    - Loss 1 : log(1 - p_negative) en excluant argmax du modèle référence (checkpoint) et GT
    - Loss 2 : log(1 - p_negative) en excluant argmax du modèle student entraîné et GT
    - À chaque nouveau meilleur Macro-F1 sur les 5000 données de validation :
      * On sauvegarde le nouveau checkpoint dans l'arborescence d'historique
      * Le meilleur modèle est chargé comme nouvelle référence (teacher/ref)
      * On continue l'entraînement avec le modèle student
    - Toutes les `patience_epochs` (30) époques consécutives sans amélioration :
      * On remonte l'arborescence des checkpoints :
        student <- checkpoint[t - 1]
        reference <- checkpoint[t - 2]
    """
    print("=" * 80)
    print(
        "PHASE 2 : Negative Learning CIFAR-10 (2000 données) sans CE + Arborescence de Rollback"
    )
    print("=" * 80)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Chargement du checkpoint initial issu de la Phase 1
    base_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    init_state = base_ckpt["model_state_dict"]
    init_f1 = float(base_ckpt["val_macro_f1"])
    init_acc = float(base_ckpt.get("val_acc", 0.0))

    student_model = build_resnet18_cifar10().to(device)
    ref_model = build_resnet18_cifar10().to(device)

    student_model.load_state_dict(copy.deepcopy(init_state))
    ref_model.load_state_dict(copy.deepcopy(init_state))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Arborescence / Historique des checkpoints de meilleurs modèles
    ckpt_0_path = ckpt_dir / "ckpt_node_000_ep000.pt"
    torch.save(
        {
            "node_idx": 0,
            "epoch": 0,
            "val_macro_f1": init_f1,
            "val_acc": init_acc,
            "model_state_dict": copy.deepcopy(init_state),
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
        student_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    print(
        f"État initial Phase 2 chargé depuis {checkpoint_path} | "
        f"BestCkpt initial: Node 0 ({best_ckpt_name}) | Val Macro-F1: {best_f1:.4f} | Val Acc: {best_acc * 100:.2f}%"
    )

    for epoch in range(1, epochs + 1):
        student_model.train()
        ref_model.eval()

        running_loss_total = 0.0
        running_loss_ref = 0.0
        running_loss_stu = 0.0
        total_samples = 0
        train_preds_list: List[torch.Tensor] = []
        train_targets_list: List[torch.Tensor] = []

        for inputs, gt_targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            gt_targets = gt_targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Prédiction du modèle référence (checkpoint)
            with torch.no_grad():
                ref_logits = ref_model(inputs)
                ref_argmax = ref_logits.argmax(dim=-1)

            # Prédiction du modèle student entraîné
            student_logits = student_model(inputs)
            with torch.no_grad():
                student_argmax = student_logits.argmax(dim=-1)

            # 1) Loss négative excluant argmax du modèle référence et GT
            loss_neg_ref = negative_unlikely_loss(
                student_logits, gt_targets, ref_argmax
            )
            # 2) Loss négative excluant argmax du modèle student entraîné et GT
            loss_neg_student = negative_unlikely_loss(
                student_logits, gt_targets, student_argmax
            )

            loss = loss_neg_ref + loss_neg_student
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss_total += loss.item() * bs
            running_loss_ref += loss_neg_ref.item() * bs
            running_loss_stu += loss_neg_student.item() * bs
            total_samples += bs
            train_preds_list.append(student_argmax.detach())
            train_targets_list.append(gt_targets.detach())

        avg_loss_total = running_loss_total / max(total_samples, 1)
        avg_loss_ref = running_loss_ref / max(total_samples, 1)
        avg_loss_stu = running_loss_stu / max(total_samples, 1)

        train_preds_cat = torch.cat(train_preds_list, dim=0)
        train_targets_cat = torch.cat(train_targets_list, dim=0)
        train_f1 = compute_macro_f1(train_preds_cat, train_targets_cat)
        train_acc = float(
            (train_preds_cat == train_targets_cat).to(torch.float32).mean().item()
        )

        # Évaluation à chaque époque sur les 5000 données de validation
        val_f1, val_acc = evaluate(student_model, val_loader, device)

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
                }
            )
            current_tip = new_node_idx
            ref_node_idx = new_node_idx
            best_node_idx = new_node_idx
            best_ckpt_name = new_ckpt_path.name

            # Charger le nouveau meilleur modèle comme référence et continuer avec le student
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
                f"Train(Loss={avg_loss_total:.4f} [Ref={avg_loss_ref:.4f}, Stu={avg_loss_stu:.4f}], F1={train_f1:.4f}, Acc={train_acc * 100:.2f}%) | "
                f"Val(F1={val_f1:.4f}, Acc={val_acc * 100:.2f}%) | "
                f"BestCkpt(Node={best_node_idx}, Ep={best_epoch:03d}, F1={best_f1:.4f}, Acc={best_acc * 100:.2f}%, file={best_ckpt_name}) | "
                f"ActiveRef=Node {ref_node_idx} * [NEW BEST]"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[Phase 2 | Ep {epoch:03d}/{epochs:03d}] "
                f"Train(Loss={avg_loss_total:.4f} [Ref={avg_loss_ref:.4f}, Stu={avg_loss_stu:.4f}], F1={train_f1:.4f}, Acc={train_acc * 100:.2f}%) | "
                f"Val(F1={val_f1:.4f}, Acc={val_acc * 100:.2f}%) | "
                f"BestCkpt(Node={best_node_idx}, Ep={best_epoch:03d}, F1={best_f1:.4f}, Acc={best_acc * 100:.2f}%, file={best_ckpt_name}) | "
                f"ActiveRef=Node {ref_node_idx} (Tip t={current_tip}) | "
                f"NoImprov={epochs_without_improvement}/{patience_epochs}"
            )

            # Toutes les 30 époques sans amélioration : rollback dans l'arborescence
            if (
                epochs_without_improvement > 0
                and epochs_without_improvement % patience_epochs == 0
            ):
                student_target_idx = max(0, current_tip - 1)
                ref_target_idx = max(0, current_tip - 2)

                student_node = checkpoint_tree[student_target_idx]
                ref_node = checkpoint_tree[ref_target_idx]

                student_ckpt = torch.load(
                    student_node["path"], map_location=device, weights_only=True
                )
                ref_ckpt = torch.load(
                    ref_node["path"], map_location=device, weights_only=True
                )

                student_model.load_state_dict(
                    copy.deepcopy(student_ckpt["model_state_dict"])
                )
                ref_model.load_state_dict(
                    copy.deepcopy(ref_ckpt["model_state_dict"])
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
        description="Entraînement CIFAR-10 : Phase 1 (CE sur 1000 données) puis Phase 2 (Negative Loss sur 2000 données avec rollback t-1/t-2)."
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
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Répertoire contenant CIFAR-10.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints_cifar10",
        help="Répertoire de sauvegarde des checkpoints.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=50,
        help="Nombre d'époques pour la Phase 1 (Cross-Entropy sur 1000 données).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=120,
        help="Nombre d'époques pour la Phase 2 (Negative Loss sur 2000 données).",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=90,
        help="Nombre d'époques sans amélioration avant de remonter l'arborescence (t-1 pour student, t-2 pour référence).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille de batch.",
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
        default=1e-6,
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
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible ! Vérifiez votre installation PyTorch/GPU."
        )

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    print(f"Utilisation du GPU : {torch.cuda.get_device_name(device)} ({device})")

    # Initialisation immédiate du contexte CUDA pour qu'il apparaisse tout de suite dans nvidia-smi
    _ = torch.zeros(1, device=device)

    loader_phase1, loader_phase2, loader_val = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        num_train1=args.num_train1,
        num_train2=args.num_train2,
        num_val=args.num_val,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    phase1_ckpt_path = args.ckpt_dir / "phase1_ce_1000.pt"

    # 1. Phase 1 : Cross-Entropy sur 1000 données CIFAR-10 + sauvegarde du checkpoint
    if args.phase in ("both", "phase1"):
        model_phase1 = build_resnet18_cifar10(num_classes=10).to(device)
        train_phase1_ce(
            model=model_phase1,
            train_loader=loader_phase1,
            val_loader=loader_val,
            device=device,
            epochs=args.phase1_epochs,
            lr=args.lr_phase1,
            weight_decay=args.weight_decay,
            save_path=phase1_ckpt_path,
        )

    # 2. Phase 2 : Rechargement du checkpoint précédent, entraînement sur 1000 + 1000 = 2000 données
    # avec uniquement les deux termes de Negative Loss et gestion de l'arborescence de checkpoints
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
            ckpt_dir=args.ckpt_dir / "phase2_tree",
        )


if __name__ == "__main__":
    main()
