import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def load_dynamics(csv_path: str) -> pd.DataFrame:
    """
    Charge le CSV produit par Record_train_dynamics.
    Retourne un DataFrame (epochs x images) avec les loss par image.
    """
    df = pd.read_csv(csv_path)
    # La colonne 'step' = epoch, le reste = images
    df = df.set_index("step").drop(columns=["epoch"], errors="ignore")
    df.index.name = "epoch"
    return df


def compute_noisy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque image, calcule 4 signaux de bruit :

    1. mean_loss      — loss moyenne sur toutes les époques
                        (élevée = le modèle n'arrive jamais à bien apprendre cette image)

    2. variance       — variance de la loss au fil des époques
                        (élevée = la loss oscille, le modèle hésite)

    3. trend          — pente de régression linéaire (loss vs epoch)
                        (positive = la loss augmente au lieu de baisser)

    4. forgetting     — nombre de fois où la loss remonte d'une époque à l'autre
                        (élevé = le modèle "oublie" cette image régulièrement)

    Retourne un DataFrame trié par score composite décroissant.
    """
    records = []
    epochs = np.arange(len(df))

    for img_name in df.columns:
        losses = df[img_name].values
        valid = ~np.isnan(losses)
        if valid.sum() < 2:
            continue
        l = losses[valid]
        e = epochs[valid]

        # 1. Loss moyenne
        mean_loss = l.mean()

        # 2. Variance
        variance = l.var()

        # 3. Tendance (pente linéaire) — positif = ça monte = mauvais
        trend = np.polyfit(e, l, 1)[0]

        # 4. Forgetting events — combien de fois loss[t] > loss[t-1]
        diffs = np.diff(l)
        forgetting = (diffs > 0).sum()

        records.append({
            "img_name": img_name,
            "mean_loss": round(mean_loss, 4),
            "variance": round(variance, 6),
            "trend": round(trend, 6),
            "forgetting": int(forgetting),
        })

    scores = pd.DataFrame(records)

    # ── Score composite (rang normalisé sur chaque métrique) ──
    for col in ["mean_loss", "variance", "trend", "forgetting"]:
        scores[f"rank_{col}"] = scores[col].rank(pct=True)

    ##### A MODIFIER EN FONCTION DES ZENVIES ######################
    scores["noisy_score"] = ( ######################
        scores["rank_mean_loss"]######################
        + scores["rank_variance"]######################
        + scores["rank_trend"]######################
        + scores["rank_forgetting"]######################
    ) / 4.0######################
    ##################################################################

    scores = scores.sort_values("noisy_score", ascending=False).reset_index(drop=True)
    return scores


def plot_top_noisy(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    images_dir: str,
    masks_dir: str,
    n: int = 10,
    img_ext: str = ".tiff",
    mask_ext: str = ".png",
    alpha: float = 0.3,
):
    """
    Pour chaque image suspecte, affiche 2 figures séparées :
      1. L'image avec le masque superposé en transparence
      2. La courbe de loss au fil des époques
    """
    top = scores.head(n)

    for _, row in top.iterrows():
        name = row["img_name"]

        # ── Image + masque overlay ─────────────────────
        img_path = Path(images_dir) / f"{name}{img_ext}"
        mask_path = Path(masks_dir) / f"{name}{mask_ext}"

        if not (img_path.exists() and mask_path.exists()):
            continue

        img_numpy = np.array(Image.open(img_path).convert("RGB"))
        mask_numpy = np.array(Image.open(mask_path).convert("L"))
        mask_bool = mask_numpy == 0

        image_rgba = np.zeros((*img_numpy.shape[:2], 4))
        image_rgba[..., :3] = img_numpy / 255.0
        image_rgba[..., 3] = alpha + (1 - alpha) * mask_bool.astype(float)

        losses = df[name].dropna().values
        stats = (f"score={row['noisy_score']:.2f}  μ={row['mean_loss']:.3f}  "
                 f"σ²={row['variance']:.4f}  trend={row['trend']:.4f}  forget={row['forgetting']}")

        fig, (ax_img, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

        ax_img.imshow(image_rgba)
        ax_img.axis("off")
        ax_img.set_title(f"{name}\n{stats}", fontsize=9)

        ax_loss.plot(losses, marker="o", markersize=3, color="tab:red")
        ax_loss.fill_between(range(len(losses)), losses, alpha=0.15, color="tab:red")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("loss curve")

        plt.tight_layout()
        plt.show()


# ── Utilisation ────────────────────────────────────────────
if __name__ == "__main__":
    N_SUSPECT = 10
    DATASET_PATH = "datasets/new_anot/datasets_new"  # ← adapte ici

    csv_path = "logs/dynamics_train_val/val_dynamics/metrics.csv"
    images_dir = f"{DATASET_PATH}/images"
    masks_dir = f"{DATASET_PATH}/GT_Object"

    df = load_dynamics(csv_path)
    scores = compute_noisy_scores(df)

    print(f"\n{'='*60}")
    print(f" Top {N_SUSPECT} images suspectes (noisy labels)")
    print(f"{'='*60}\n")
    print(scores.head(N_SUSPECT).to_string(index=False))

    plot_top_noisy(df, scores, images_dir, masks_dir, n=N_SUSPECT)

    scores.to_csv("noisy_label_scores.csv", index=False)
    print(f"\nScores sauvegardés dans noisy_label_scores.csv")