import argparse

from pytorch_lightning import Trainer, seed_everything

from dataset_interface.dataset_real import ParkingDataModuleReal
from utils.config import get_train_config_obj
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import time
from loss.traj_point_loss import TokenTrajPointLoss
from loss.traj_point_loss import TrajPointLoss
from utils.eval import get_eval_metric_results
from model_interface.model.parking_model_real import ParkingModelReal
from torch.utils.data import random_split
from ruamel.yaml import YAML
from inference import test_main
from model_interface.model.network import AE_Conv
import torch.nn.functional as F


# 一些默认超参数，可以按需改
ae_lr         = 1e-3
ae_epochs     = 200
val_every      = 5
show_every_step = 50
save_dir       = "./trained_params"
date_record    = "imgae2511"  # 方便区分模型

def save_ae_checkpoint(checkpoint_dir, ae_model, optimizer, epoch, val_loss, date):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "state_dict": ae_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    ckpt_path = os.path.join(
        checkpoint_dir,
        f"ae_epoch_{epoch}.valloss_{val_loss:.4f}.{date}.pth"
    )
    torch.save(state, ckpt_path)
    print(f"[AE] model checkpoint saved to: {ckpt_path}")
    
def train_img_ae(config_obj):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    full_dataset = ParkingDataModuleReal(config_obj, is_train=1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

# 随机分割数据集
    dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(dataset_train, batch_size=config_obj.batch_size, shuffle=True, num_workers=config_obj.num_workers)
    val_loader = DataLoader(dataset_val, batch_size= config_obj.batch_size, shuffle=False, num_workers=config_obj.num_workers)

      # ========= 2. 构建 AE_Conv =========
    # 这里假设你的 config 里已经有下面这些字段：
    #   img_shape        : (C, H, W)
    #   k_img_conv       : 卷积核 K
    #   img_conv_layers  : 通道数列表，例如 [16, 32, 64]
    #   img_linear_layers: 全连接层大小列表，例如 [256, 128]
    #   embed_size       : 潜变量维度(也是 ImgEncoder 输出维度)
    img_shape        = config_obj.img_shape
    k_img_conv       = config_obj.k_img_conv
    c_conv_list      = config_obj.img_conv_layers
    size_fc_list     = config_obj.img_linear_layers
    embed_size       = config_obj.embed_size

    ae = AE_Conv(
        img_shape=img_shape,
        k=k_img_conv,
        embed_size=embed_size,
        c_conv_list=c_conv_list,
        size_fc_list=size_fc_list,
        use_tanh=False,   # 和 network.py 保持一致，一般用 LeakyReLU
    ).to(device)

    
    optimizer = optim.Adam(ae.parameters(), lr=ae_lr, weight_decay=0)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    # training loop
    best_val_loss = float("inf")
    global_step = 0
 
    # ========= 3. 训练循环 =========
    for epoch in range(ae_epochs):
        ae.train()
        epoch_loss = 0.0
        num_samples = 0
        start_tic = time.time()

        for batch in train_loader:
            # batch 可能是 torch_geometric.data.Data / Batch，也可能是 dict
            if hasattr(batch, "image"):
                imgs = batch.image
            else:
                imgs = batch["image"]

            imgs = imgs.to(device)

            optimizer.zero_grad()

            # AE 前向
            recon_x = ae(imgs)

            # 重构损失（也可以用 BCE，根据你图像是否 0~1）
            recon_loss = F.mse_loss(recon_x, imgs, reduction="mean")


            loss = recon_loss 
            loss.backward()
            optimizer.step()

            B = imgs.size(0)
            epoch_loss += loss.item() * B
            num_samples += B
            global_step += 1

            if global_step % show_every_step == 0:
                print(
                    f"[AE][epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.6f} "
                    f"(recon={recon_loss.item():.6f}), "
                    f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, num_samples)
        print(
            f"[AE] Epoch {epoch} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.6f} | "
            f"time={time.time() - start_tic:.2f}s"
        )

     # ========= 4. 简单的验证 =========
        if (epoch + 1) % val_every == 0:
            ae.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    if hasattr(batch, "img"):
                        imgs = batch.img
                    else:
                        imgs = batch["img"]
                    imgs = imgs.to(device)

                    recon_x = ae(imgs)
                    recon_loss = F.mse_loss(recon_x, imgs, reduction="mean")
                    loss = recon_loss
                    B = imgs.size(0)
                    al_loss += loss.item() * B
                    al_samples += B

            avg_val_loss = val_loss / max(1, val_samples)
            print(f"[AE] Epoch {epoch} | val_loss={avg_val_loss:.6f}")

            # 保存最优 checkpoint（用于调试/恢复）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_ae_checkpoint(save_dir, ae, optimizer, epoch, best_val_loss, date_record)

    # ========= 5. 训练结束后，保存完整的 AE 模型，用于后续加载 ImgEncoder =========
    final_path = os.path.join(save_dir, f"ae_final.{date_record}.pth")
    # 使用 AE_Conv 自带的 save，保持和 MultiObsEmbedding.load_img_encoder 兼容
    ae.save(final_path)   # 内部是 torch.save(self, path)
    print(f"[AE] final model saved to: {final_path}")

def main():
    seed_everything(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = arg_parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    train_img_ae(config_obj)


if __name__ == '__main__':
    main()