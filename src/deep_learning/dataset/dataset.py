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

import torch.nn.functional as F
import torch
import lightning as L
from torchvision.transforms import ToTensor, RandomVerticalFlip, RandomHorizontalFlip, Compose, Normalize
from torch.utils.data import DataLoader, Dataset

import numpy as np

class DataSet512Mask(Dataset):


    def __init__(self, images_paths, masks_paths, transform=None, keep_pourcent = 1):

        self.transform = ToTensor()
        lambda_transform = lambda x: ((x-1)*-1).abs()
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_image = Compose([self.transform, normalize])
        self.transform_mask = Compose([self.transform, lambda_transform])
        self.images = sorted(images_paths)
        self.masks = sorted(masks_paths)
        if keep_pourcent < 1:
            keep_size = int(len(self.images)*keep_pourcent)
            random_idx = np.random.choice(len(self.images), keep_size, replace=False)
            self.images = [self.images[i] for i in random_idx]
            self.masks = [self.masks[i] for i in random_idx]
        
        self.img_names = [os.path.basename(p).split(".")[0] for p in self.images]

       

        self.augment_2 = Compose([RandomHorizontalFlip(p=1.0)])
        self.augment_3 = Compose([RandomVerticalFlip(p=1.0)])
        self.augment_1 = Compose([RandomHorizontalFlip(p=1.0), RandomVerticalFlip(p=1.0)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = self.transform_image(Image.open(self.images[idx]).convert("RGB"))
        mask = self.transform_mask(Image.open(self.masks[idx]).convert("L"))
        P= torch.rand(1)
        if P < 0.33:
            image, mask = self.augment_2(image), self.augment_2(mask)
        elif P < 0.66:
            image, mask = self.augment_3(image), self.augment_3(mask)
        else :
            image, mask = self.augment_1(image), self.augment_1(mask)
        return {"image":image, "mask":mask, "idx":idx, "img_name":self.img_names[idx]}
    
    def plot(self,idx):
        import matplotlib.pyplot as plt
        img = self.images[idx].permute(1,2,0).numpy()
        mask = self.masks[idx].squeeze().numpy()
        values, counts = np.unique(mask, return_counts=True)
        print(dict(zip(values, counts)))

        f,axes = plt.subplots(1, 3)
        axes[0].imshow(img )
        axes[1].imshow(mask , cmap='gray')
        alpha_img = np.zeros(shape=img.shape[0:2] + (4,))
        alpha_img[...,0:3] = img
        alpha_img[...,3] = 0.3 + 0.7*mask 
        axes[2].imshow(alpha_img )
        plt.show()
    

class DataModule512Mask(L.LightningDataModule):

    def __init__(self, dataset_path, batch_size=8, num_workers=4, keep_pourcent=1, val_split=0.2):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.keep_pourcent = keep_pourcent,
        self.val_split = val_split

    def setup(self, stage=None):

        images_files = np.array(sorted(glob.glob(os.path.join(self.dataset_path, "images", "*.tiff"))))
        masks_files = np.array(sorted(glob.glob(os.path.join(self.dataset_path, "GT_Object", "*.png"))))

        np.random.seed(42)
        # train_idx = np.random.choice(len(images_files), int((1 - self.val_split)*len(images_files)), replace=False)
        # val_idx = np.array([i for i in range(len(images_files)) if i not in train_idx])

        # train_idx_2_pourcent = np.random.choice(train_idx, max(10, int(0.000005*len(train_idx))), replace=False)
        # train_idx = train_idx_2_pourcent

        # val_idx_2_pourcent = np.random.choice(val_idx, max(12, int(0.000005*len(val_idx))), replace=False)
        # val_idx = val_idx_2_pourcent

        # train_images = images_files[train_idx]
        # train_masks = masks_files[train_idx]
        # val_images = images_files[val_idx]
        # val_masks = masks_files[val_idx]

        self.train_dataset = DataSet512Mask(
            images_paths=images_files,
            masks_paths=masks_files,
            keep_pourcent=1
        )
        self.val_dataset = DataSet512Mask(
            images_paths=images_files,
            masks_paths=masks_files,
            keep_pourcent=1
        )
        

        print(f"Train dataset size: {len(self.train_dataset)}")

        print(f"Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)



def compute_global_pos_weight(dataloader):  # dataloader de train
    pos = 0
    neg = 0
    with torch.no_grad():
        for batch in dataloader:
            y = batch["mask"] if isinstance(batch, dict) else batch[1]  # (B,1,H,W) in {0,1}
            pos += y.sum().item()
            neg += (1 - y).sum().item()
    # éviter division par zéro
    pos = max(pos, 1.0)
    return neg / pos


if __name__ == "__main__":
    dataset_path = "/home/auguste/Desktop/Cute_Track/datasets/Train_005"
    data_module = DataModule512Mask(dataset_path, batch_size=4)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    pw = compute_global_pos_weight(train_loader)
    print(f"Global pos weight: {pw}")