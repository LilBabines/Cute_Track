"""
predict_seg.py
==============
Usage:
    python predict_seg.py \
        --images   /path/to/images \
        --output   /path/to/output \
        --ckpt     /path/to/model.ckpt \
        --gt       /path/to/ground_truth   # optionnel

Produit pour chaque image une figure côte-à-côte :
  [Image | Prediction | (Ground Truth)]  sauvegardée en PNG dans --output.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- à adapter à ton projet ----------
from src.deep_learning.models.SAMUNET import LitBinarySeg, SAM2UNet

# ---------------------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ── Transforms (même normalisation qu'à l'entraînement) ──────────────
def get_inference_transform(size: tuple[int, int] = (512, 512)):
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Charge un masque GT, resize et binarise."""
    m = Image.open(path).convert("L").resize(size[::-1], Image.NEAREST)
    m = np.array(m, dtype=np.float32)
    return (m > 127).astype(np.float32)


# ── Visualisation ────────────────────────────────────────────────────
OVERLAY_ALPHA = 0.45
PRED_COLOR = np.array([1.0, 0.2, 0.2])   # rouge
GT_COLOR   = np.array([0.2, 1.0, 0.2])   # vert

def overlay(img_np: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha=OVERLAY_ALPHA):
    """Superpose un masque binaire coloré sur une image RGB [0-1]."""
    out = img_np.copy()
    roi = mask > 0.5
    out[roi] = out[roi] * (1 - alpha) + color * alpha
    return out

def save_figure(img_np, pred_mask, gt_mask, save_path: Path, threshold: float):
    has_gt = gt_mask is not None
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # --- image originale ---
    axes[0].imshow(img_np)
    axes[0].set_title("Image", fontsize=13)
    axes[0].axis("off")

    # --- prédiction overlay ---
    pred_vis = overlay(img_np, pred_mask, PRED_COLOR)
    axes[1].imshow(pred_vis)
    axes[1].set_title(f"Prediction (thr={threshold})", fontsize=13)
    axes[1].axis("off")
    pred_patch = mpatches.Patch(color=PRED_COLOR, label="Pred")
    axes[1].legend(handles=[pred_patch], loc="lower right", fontsize=10)

    # --- ground truth overlay ---
    if has_gt:
        gt_vis = overlay(img_np, gt_mask, GT_COLOR)
        axes[2].imshow(gt_vis)
        axes[2].set_title("Ground Truth", fontsize=13)
        axes[2].axis("off")
        gt_patch = mpatches.Patch(color=GT_COLOR, label="GT")
        axes[2].legend(handles=[gt_patch], loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Inférence ────────────────────────────────────────────────────────
@torch.no_grad()
def predict_folder(
    ckpt_path: str,
    images_dir: str,
    output_dir: str,
    gt_dir: str | None = None,
    img_size: tuple[int, int] = (512, 512),
    threshold: float = 0.5,
    device: str = "cuda",
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(gt_dir) if gt_dir else None

    # ── Chargement modèle ──
    net = SAM2UNet(config="tiny", sam_checkpoint_path="src/sam2/checkpoints/sam2.1_hiera_tiny.pt", freeze_encorder = True).to("cuda")
    model = LitBinarySeg.load_from_checkpoint(ckpt_path, net=net, map_location=device)
    model.eval().to(device)

    transform = get_inference_transform(img_size)

    # ── Liste des images ──
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        print(f"Aucune image trouvée dans {images_dir}")
        return

    print(f"{'─'*60}")
    print(f"  Images   : {images_dir}  ({len(img_paths)} fichiers)")
    print(f"  Output   : {output_dir}")
    print(f"  GT       : {gt_dir or 'non spécifié'}")
    print(f"  Device   : {device}")
    print(f"  Threshold: {threshold}")
    print(f"{'─'*60}")

    dice_scores = []

    for img_path in img_paths:
        # --- charger et transformer l'image ---
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

        # --- inférence ---
        logits = model(img_tensor)                                # (1,1,H,W)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()      # (H,W)
        pred_mask = (prob >= threshold).astype(np.float32)

        # --- image numpy pour affichage (resize à img_size) ---
        img_display = np.array(pil_img.resize(img_size[::-1], Image.BILINEAR)) / 255.0

        # --- ground truth (optionnel) ---
        gt_mask = None
        if gt_dir:
            # tente de trouver le fichier GT avec le même nom (ou .png)
            gt_candidates = [
                gt_dir / img_path.name,
                gt_dir / img_path.with_suffix(".png").name,
                gt_dir / img_path.with_suffix(".tif").name,
            ]
            gt_file = next((c for c in gt_candidates if c.exists()), None)
            if gt_file:
                gt_mask = load_mask(gt_file, img_size)
                # Dice rapide
                inter = (pred_mask * gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum()
                dice = (2 * inter / (union + 1e-6)) if union > 0 else 1.0
                dice_scores.append(dice)

        # --- sauvegarde figure ---
        save_name = img_path.stem + "_pred.png"
        save_figure(img_display, pred_mask, gt_mask, output_dir / save_name, threshold)

        dice_str = f"  dice={dice_scores[-1]:.4f}" if gt_mask is not None else ""
        print(f"  ✓ {img_path.name}{dice_str}")

    # --- résumé ---
    print(f"{'─'*60}")
    print(f"  {len(img_paths)} images traitées → {output_dir}")
    if dice_scores:
        mean_dice = np.mean(dice_scores)
        print(f"  Dice moyen (sur {len(dice_scores)} images avec GT) : {mean_dice:.4f}")
    print(f"{'─'*60}")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prédiction segmentation binaire")
    parser.add_argument("--images", required=True, help="Dossier d'images d'entrée")
    parser.add_argument("--output", required=True, help="Dossier de sortie")
    parser.add_argument("--gt", default=None, help="Dossier ground truth (optionnel)")
    parser.add_argument("--ckpt", required=True, help="Chemin du checkpoint (.ckpt)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[512, 512], help="H W")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    predict_folder(
        ckpt_path=args.ckpt,
        images_dir=args.images,
        output_dir=args.output,
        gt_dir=args.gt,
        img_size=tuple(args.img_size),
        threshold=args.threshold,
        device=args.device,
    )

if __name__ == "__main__":
    main()
