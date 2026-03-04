import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2  # pip install opencv-python
import os
from tqdm import tqdm
import shutil

def json_to_mask(json_path, output_path=None):
    """
    Convertit un fichier JSON LabelMe en masque PNG.
    Fond = 255 (blanc), polygones = 0 (noir).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.full((h, w), 0, dtype=np.uint8)

    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)

    if output_path is None:
        output_path = Path(json_path).with_suffix(".png")

    Image.fromarray(mask).save(output_path)


def copy_image(json_path, image_dir, output_dir):
    """
    Copie l'image source référencée dans le JSON vers output_dir.
    """
    src = Path(image_dir) / f"{Path(json_path).stem}.tiff"
    ward = Path(json_path).parent.name
    assert src.exists(), f"Image introuvable : {src}"
    dst = Path(output_dir) / f"{ward}_{src.name}"
    shutil.copy2(src, dst)


# --- Traitement en batch ---
if __name__ == "__main__":
    dirs = ["bokonomela", "bupu", "images_dutumi_ward_tza", "images_fukayose_ward_tza"]

    json_files = []
    for d in dirs:
        for ward in os.listdir(f"datasets/new_anot/{d}/17"):
            p = Path(f"datasets/new_anot/{d}/17/{ward}")
            assert p.exists(), f"Dossier introuvable : {p}"
            json_files.extend(sorted(p.glob("*.json")))

    outdir_mask = "/home/auguste/Desktop/Cute_Track/datasets/new_anot/datasets_new/GT_Object"
    outdir_img = "/home/auguste/Desktop/Cute_Track/datasets/new_anot/datasets_new/images"
    output_json = "/home/auguste/Desktop/Cute_Track/datasets/new_anot/datasets_new/json"

    Path(outdir_mask).mkdir(parents=True, exist_ok=True)
    Path(outdir_img).mkdir(parents=True, exist_ok=True)
    Path(output_json).mkdir(parents=True, exist_ok=True)

    for jf in tqdm(json_files):
        ward = jf.parent.name
        out = Path(outdir_mask) / f"{ward}_{jf.stem}.png"
        dst = Path(output_json) / f"{ward}_{jf.stem}.json"

        shutil.copy2(jf, dst)
        # json_to_mask(jf, output_path=out)
        # copy_image(jf, jf.parent, outdir_img)

    print(f"\nTerminé — {len(json_files)} masque(s) généré(s).")