from src.deep_learning.models.SAMUNET import LitBinarySeg, SAM2UNet
from src.deep_learning.models.SAMUNext import SAM2UNeXT
from lightning.pytorch.callbacks import ModelCheckpoint
from src.deep_learning.utils_dl import Record_train_dynamics
from src.deep_learning.dataset.dataset import DataModule512Mask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.segmentation import DiceScore
import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    exp_name = "dynamics_train_val"

    #define callbacks
    cb_checkpoint = ModelCheckpoint(
        monitor="val/dice",
        dirpath=f"logs/{exp_name}/",
        save_top_k=1,
        mode="max",
    )
    record_D = Record_train_dynamics(save_dir="logs/", name=exp_name)

    #initialize model, litmodule, trainer, datamodule
    net = SAM2UNet(config="tiny", sam_checkpoint_path="src/sam2/checkpoints/sam2.1_hiera_tiny.pt", freeze_encorder = False).to("cuda")
    # net = SAM2UNeXT(checkpoint_path="models/sam2_hiera_large.pt",dinov2_path="models/model.safetensors").to("cuda")

    lit = LitBinarySeg(
        net,
        deep_supervision=False,
        dice_use_all_outputs=False,      # mets True si tu veux la Dice moyenne (out,out1,out2)
        pos_weight=30                   # utile si classe rare
    )
    trainer = Trainer(accelerator="auto", devices=1, 
                      max_epochs=500,
                      log_every_n_steps=10,
                      logger=TensorBoardLogger(save_dir="logs/", name=exp_name, version="tensorboard"),
                      num_sanity_val_steps = 0, 
                      callbacks=[cb_checkpoint, record_D])# 
    
    data_module = DataModule512Mask(
        dataset_path="datasets/new_anot/datasets_new",
        batch_size=16,
        num_workers=8,
        keep_pourcent=1
    )

    trainer.fit(lit, datamodule=data_module)

    best_model_path = cb_checkpoint.best_model_path
    print(f"Best model path: {best_model_path}")

    # module = LitBinarySeg.load_from_checkpoint(best_model_path,net=net).to("cuda")
    # lit.eval()
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # 2) Récupérer un batch de la val
    # batch = next(iter(data_module.val_dataloader()))
    # if isinstance(batch, dict):
    #     img, mask = batch["image"], batch["mask"]
    # else:
    #     img, mask = batch  # attendu: (B,C,H,W), (B,1,H,W) in {0,1}

    # img = img.to(device, non_blocking=True)
    # mask = mask.to(device, non_blocking=True).float()

    # # On va regarder le premier élément du batch
    # idx = 0
    # with torch.no_grad():
    #     # forward retourne les logits (B,1,H,W) via .forward() de LitBinarySeg
    #     logits = lit(img)                # (B,1,H,W)
    #     probs  = torch.sigmoid(logits)      # proba foreground
    #     p      = probs[idx:idx+1]           # (1,1,H,W)
    #     y      = mask[idx:idx+1]            # (1,1,H,W)

    # # 3) Calcul Dice (foreground uniquement) avec DiceScore (one-hot 2 canaux)
    # metric = DiceScore(num_classes=2, include_background=False, input_format="one-hot").to(device)

    # preds2 = torch.cat([1.0 - p, p], dim=1)   # (1,2,H,W) => [bg, fg]
    # tgt2   = torch.cat([1.0 - y, y], dim=1)   # (1,2,H,W)

    # with torch.no_grad():
    #     dice_val = metric(preds2, tgt2).item()

    # print(f"Dice (fg only) = {dice_val:.4f}")

    # # 4) Affichage matplotlib : image / GT / prédiction binaire
    # thr = 0.5  # ajuste si tu as choisi un seuil optimisé sur la val
    # pred_bin = (p > thr).float().squeeze().detach().cpu()   # (H,W)
    # gt       = y.squeeze().detach().cpu()                   # (H,W)

    # # préparation image pour affichage
    # im = img[idx].detach().cpu()  # (C,H,W)
    # plt.figure(figsize=(12,4))

    # plt.subplot(1,3,1)
    # if im.ndim == 3 and im.shape[0] == 3:
    #     plt.imshow(im.permute(1,2,0).clamp(0,1))
    # elif im.ndim == 3 and im.shape[0] == 1:
    #     plt.imshow(im.squeeze(0), cmap="gray", vmin=0, vmax=1)
    # else:
    #     # fallback
    #     plt.imshow(im.permute(1,2,0))
    # plt.title("Image")
    # plt.axis("off")

    # plt.subplot(1,3,2)
    # plt.imshow(gt, cmap="gray")
    # plt.title("Mask GT")
    # plt.axis("off")

    # plt.subplot(1,3,3)
    # plt.imshow(pred_bin, cmap="gray")
    # plt.title(f"Pred @ thr={thr}")
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()
