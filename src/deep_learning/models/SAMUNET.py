# SAM-UNet adaptation
# @article{xiong2024sam2,
#   title={SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation},
#   author={Xiong, Xinyu and Wu, Zihuang and Tan, Shuangyi and Li, Wenxue and Tang, Feilong and Chen, Ying and Li, Siying and Ma, Jie and Li, Guanbin},
#   journal={arXiv preprint arXiv:2408.08870},
#   year={2024}
# } https://github.com/WZH0120/SAM2-UNet/tree/main


import glob
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import lightning as L
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import numpy as np
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

def structure_loss(pred, mask):

    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    



class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    



class SAM2UNet(nn.Module):
    def __init__(self, config= "small", pl_sam2Unet_checkpoint_path = None,sam_checkpoint_path=None, freeze_encorder = False) -> None:
        super(SAM2UNet, self).__init__()    
        self.size = config
        if config in ["small", "tiny", "large"]:
            sam_config = f"sam2_hiera_{config[0]}.yaml"
        elif config == "base":
            sam_config = "sam2_hiera_b+.yaml"
        else:
            raise ValueError(f"Unknown config {config}", "expected small, tiny, base, large")
        if sam_checkpoint_path and not pl_sam2Unet_checkpoint_path:
            model = build_sam2(sam_config, sam_checkpoint_path)
        else:
            model = build_sam2(sam_config)
        
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        if pl_sam2Unet_checkpoint_path:
                checkpoint = torch.load(pl_sam2Unet_checkpoint_path)
                net_weights = {k[4:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("net.")}
                model.load_state_dict(net_weights)

        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encorder
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        RFB_modified_input_channels = {"small": [96, 192, 384, 768],
                                       "tiny" : [96, 192, 384, 768],
                                        "base" : [112, 224, 448, 896],
                                       "large" : [144, 288, 576, 1152]}
        
        rfb1_in, rfb2_in, rfb3_in, rfb4_in = RFB_modified_input_channels[config]
        self.rfb1 = RFB_modified(rfb1_in, 64)
        self.rfb2 = RFB_modified(rfb2_in, 64)
        self.rfb3 = RFB_modified(rfb3_in, 64)
        self.rfb4 = RFB_modified(rfb4_in, 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)


        self.encoder.train()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')

        return out, out1, out2

def _to_onehot2_from_logits(logits, target_01):
    preds_hard = (torch.sigmoid(logits) > 0.5).float()
    preds2 = torch.cat([1.0 - preds_hard, preds_hard], dim=1)
    tgt2   = torch.cat([1.0 - target_01, target_01], dim=1)
    return preds2, tgt2


class LitBinarySeg(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        deep_supervision: bool = False,
        aux_weights: tuple[float, float] = (0.3, 0.3),  # (out1, out2) dans la loss
        pos_weight: float | None = 800,                # pour BCEWLL si classe rare
        use_scheduler: bool = True,
        dice_weight: float = 1.0,
        dice_use_all_outputs: bool = False,             # True => log step moyen (out,out1,out2)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

        # --- loss ---
        if pos_weight is not None:
            self.register_buffer("pos_w_buf", torch.tensor([pos_weight]), persistent=False)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_w_buf, reduction="none")
        else:
            self.pos_w_buf = None
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # --- metrics (état par split) ---
        # binaire => 2 classes, on ignore le background pour le score
        metric_kwargs = dict(num_classes=2, include_background=False, input_format="one-hot", average="micro")
        self.train_dice = DiceScore(**metric_kwargs)
        self.val_dice   = DiceScore(**metric_kwargs)
        self.test_dice  = DiceScore(**metric_kwargs)

        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_precision", BinaryPrecision())
            setattr(self, f"{stage}_recall",    BinaryRecall())
            setattr(self, f"{stage}_f1",        BinaryF1Score())


    # -------- utils
    def _dice_loss(self, logits, target, eps=1e-6):
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        dice = (2 * inter + eps) / den
        loss = 1 - dice

        # ignorer les images sans positif (masque vide)
        has_pos = target.sum(dim=(2, 3)) > 0
        if has_pos.any():
            return loss[has_pos].mean()
        else:
            return torch.tensor(0.0, device=logits.device)
    
    def _update_prf_metrics(self, stage: str, logits, target):
        """Met à jour Precision, Recall, F1 pour le split donné."""
        preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
        tgt = target.long().view(-1)
        getattr(self, f"{stage}_precision").update(preds, tgt)
        getattr(self, f"{stage}_recall").update(preds, tgt)
        getattr(self, f"{stage}_f1").update(preds, tgt)

    def _log_and_reset_prf(self, stage: str):
        """Compute, log et reset en fin d'époque."""
        for name in ("precision", "recall", "f1"):
            metric = getattr(self, f"{stage}_{name}")
            self.log(f"{stage}/{name}", metric.compute(), prog_bar=(name == "f1"), on_step=False, on_epoch=True)
            metric.reset()

    def _compute_losses(self, preds_tuple, target):
        out, out1, out2 = preds_tuple

        bce_main_no_mean = self.bce_loss(out, target).mean(dim=(2,3))

        bce_main = bce_main_no_mean.mean()
        
        dice_main_no_mean = self._dice_loss(out, target)

        dice_main = dice_main_no_mean
        loss_main = bce_main + self.hparams.dice_weight * dice_main
        loss_main_no_mean = bce_main_no_mean + self.hparams.dice_weight * dice_main_no_mean

        losses = {"loss/main": loss_main, "loss/bce": bce_main, "loss/dice": dice_main}

        if self.hparams.deep_supervision:
            w1, w2 = self.hparams.aux_weights
            tgt1 = F.interpolate(target, size=out1.shape[-2:], mode="nearest")
            tgt2 = F.interpolate(target, size=out2.shape[-2:], mode="nearest")
            bce1 = self.bce_loss(out1, tgt1).mean(dim=(2,3)).mean()
            bce2 = self.bce_loss(out2, tgt2).mean(dim=(2,3)).mean()
            dice1 = self._dice_loss(out1, tgt1)
            dice2 = self._dice_loss(out2, tgt2)
            loss1 = bce1 + self.hparams.dice_weight * dice1
            loss2 = bce2 + self.hparams.dice_weight * dice2

            losses["loss/aux1"] = loss1
            losses["loss/aux2"] = loss2
            loss_total = loss_main + w1 * loss1 + w2 * loss2

        else:
            loss_total = loss_main

        losses["loss/total"] = loss_total
        return loss_total, losses, loss_main_no_mean

    def _update_dice_epoch_metric(self, stage: str, out_logits, target):
        """met à jour la metric d'époque avec la sortie principale uniquement"""
        preds2, tgt2 = _to_onehot2_from_logits(out_logits, target)
        if stage == "train":
            self.train_dice.update(preds2, tgt2)
        elif stage == "val":
            self.val_dice.update(preds2, tgt2)
        else:
            self.test_dice.update(preds2, tgt2)

    def _log_step_dice_mean_if_needed(self, stage: str, preds_tuple, target):
        """optionnel: log step d'une moyenne (out,out1,out2) sans toucher à l'état d'époque"""
        if not self.hparams.dice_use_all_outputs:
            return
        out, out1, out2 = preds_tuple
        # on crée une metric éphémère pour ce step (pas d'état cumulé)
        tmp_metric = DiceScore(num_classes=2, include_background=False, input_format="one-hot", average="micro").to(out.device)

        def batch_dice(logits, tgt):
            p2, t2 = _to_onehot2_from_logits(logits, tgt)
            return tmp_metric(p2, t2)

        # redimensionner les masques pour les sorties auxiliaires
        tgt1 = F.interpolate(target, size=out1.shape[-2:], mode="nearest")
        tgt2 = F.interpolate(target, size=out2.shape[-2:], mode="nearest")

        d_main = batch_dice(out,  target)
        d1     = batch_dice(out1, tgt1)
        d2     = batch_dice(out2, tgt2)
        d_mean = (d_main + d1 + d2) / 3.0
        self.log(f"{stage}/dice_step_mean3", d_mean, prog_bar=False, on_step=True, on_epoch=False)

    # -------- steps
    def _step(self, batch, stage: str):
        x, y = (batch["image"], batch["mask"]) if isinstance(batch, dict) else batch[0:2]  # y: (B,1,H,W) in {0,1} , add idx in get_item be careful
        preds = self.net(x)  # (out, out1, out2) logits

        loss, losses_dict, losses_not_mean = self._compute_losses(preds, y)
        self._update_dice_epoch_metric(stage, preds[0], y)
        self._update_prf_metrics(stage, preds[0], y)
        # self._log_step_dice_mean_if_needed(stage, preds, y)

        # logs des pertes
        for k, v in losses_dict.items():
            self.log(f"{stage}/{k}", v, prog_bar=(k == "loss/total"), on_step=(stage=="train"), on_epoch=True, batch_size=x.shape[0])
        return {"loss": loss, "all_losses": losses_not_mean}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_train_epoch_end(self):
        dice = self.train_dice.compute()
        self.log("train/dice", dice, prog_bar=True, on_step=False, on_epoch=True)
        self._log_and_reset_prf("train")
        self.train_dice.reset()

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_validation_epoch_end(self):
        dice = self.val_dice.compute()
        self.log("val/dice", dice, prog_bar=True, on_step=False, on_epoch=True)
        self._log_and_reset_prf("val")
        self.val_dice.reset()

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_test_epoch_end(self):
        dice = self.test_dice.compute()
        self.log("test/dice", dice, prog_bar=True, on_step=False, on_epoch=True)
        self.test_dice.reset()

    def forward(self, x):
        out, _, _ = self.net(x)
        return out  # logits (B,1,H,W)

    def configure_optimizers(self):
        encoder_params = list(self.net.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        decoder_params = [p for p in self.net.parameters() if id(p) not in encoder_ids and p.requires_grad]

        opt = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.hparams.lr},          # 1e-5
            {"params": decoder_params, "lr": self.hparams.lr * 10},     # 1e-4
        ], weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs if self.trainer is not None else 100
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    def setup(self, stage: str):
        if self.pos_w_buf is not None and self.pos_w_buf.device != self.device:
            self.bce_loss.pos_weight = self.pos_w_buf.to(self.device)