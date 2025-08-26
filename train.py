import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_interface.dataloader import ParkingDataloaderModule
from dataset_interface.dataset_real import ParkingDataModuleReal
from utils.config import get_train_config_obj
from dataset import GraphData
from dataset import GraphDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import time
from loss.traj_point_loss import TokenTrajPointLoss
from loss.traj_point_loss import TrajPointLoss
from utils.eval import get_eval_metric_results
from model_interface.model.parking_model_real import ParkingModelReal
from torch_geometric.data import Batch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import random_split
from ruamel.yaml import YAML
from inference import test_main


decay_lr_factor = 0.3
decay_lr_every = 10
lr = 0.0001
epochs = 200
end_epoch = 0
lr = 0.0001
show_every = 20
val_every = 5
best_minade = float('inf')
save_dir = './trained_params'
date_record = "250825"
global_step = 0

class MinStepLR(optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1):
        self.min_lr = min_lr
        super(MinStepLR, self).__init__(optimizer, step_size, gamma, last_epoch)
    
    def get_lr(self):
        lrs = super(MinStepLR, self).get_lr()
        # 确保学习率不低于最小值
        return [max(lr, self.min_lr) for lr in lrs]
def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        'val_minade': val_minade
        }
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"ParkE2E"}.pth')
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    return checkpoint_path['end_epoch']



def collate_graph(batch_list):
    """
    batch_list: List[GraphData]
    返回一张大图，额外字段仍是 batch-level 的 tensor
    """
    # 用 Batch 把小图拼成大图
    big_graph = Batch.from_data_list(batch_list)

    # 其余字段用默认 collate 拼出 [B, ...]
    # tensor_keys = ['gt_traj_point', 'gt_traj_point_token',
    #                'target_point', 'fuzzy_target_point']
    tensor_keys = ['gt_traj_point', 'gt_traj_point_token']
    tensor_dict = default_collate([{k: getattr(g, k) for k in tensor_keys}
                                   for g in batch_list])

    # 把张量挂到大图上
    for k, v in tensor_dict.items():
        setattr(big_graph, k, v)

    return big_graph
def train(config_obj):
    global global_step
    global best_minade
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    # dataset_train = ParkingDataModuleReal(config_obj, is_train=1)
    # dataset_val = ParkingDataModuleReal(config_obj, is_train=0)
    full_dataset = ParkingDataModuleReal(config_obj, is_train=1)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

# 随机分割数据集
    dataset_train, dataset_val = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(dataset_train, batch_size=config_obj.batch_size, shuffle=True, num_workers=config_obj.num_workers, collate_fn=collate_graph)
    val_loader = DataLoader(dataset_val, batch_size= config_obj.batch_size, shuffle=False, num_workers=config_obj.num_workers, collate_fn=collate_graph)

    max_id = 0
    for g in full_dataset.graph_dataset:
        max_id = max(max_id, g.cluster.max().item())
    # for g in dataset_val.graph_dataset:
    #     max_id = max(max_id, g.cluster.max().item())
    config_obj.max_id = max_id
    yaml = YAML()
    yaml.preserve_quotes = True    
    yaml.width = 4096               

    with open("./config/training_real.yaml", "r", encoding="utf-8") as f:
        cfg_dict = yaml.load(f)
    cfg_dict["max_id"] = int(max_id)
    with open("./config/training_real.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    model = ParkingModelReal(config_obj)
    model = model.to(device=device)
    traj_point_loss_func = None
    if config_obj.decoder_method == "transformer":
        traj_point_loss_func = TokenTrajPointLoss(config_obj)
    elif config_obj.decoder_method == "gru":
        traj_point_loss_func = TrajPointLoss(config_obj)

    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # scheduler = MinStepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor,min_lr=1e-5)
    


    # training loop
    model.train()
    for epoch in range(epochs):
        print(epoch)
        acc_loss = .0
        num_samples = 0
        start_tic = time.time()
        for data in train_loader:
            # for key, val in data.items():
            #     if isinstance(val, torch.Tensor):
            #         data[key] = val.to(device)
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = traj_point_loss_func(out, data, global_step)
            loss.backward()
            acc_loss += config_obj.batch_size * loss.item()
            num_samples += data["gt_traj_point_token"].shape[0]
            optimizer.step()
            global_step += 1
            if (global_step + 1) % show_every == 0:
                print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        scheduler.step()
        print(
            f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
        if (epoch+1) % val_every == 0 and (not epoch < end_epoch):
            print("eval as epoch:{epoch}")
            metrics = get_eval_metric_results(config_obj, model, val_loader, device, 19)
            curr_minade = metrics
            print(f"minADE:{metrics:3f}")

            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date_record)
        model.train()
                
    # eval result on the identity dataset
    # metrics = get_eval_metric_results(config_obj, model, val_loader, device)
    # curr_minade = metrics
    # if curr_minade < best_minade:
    #     best_minade = curr_minade
    save_checkpoint(save_dir, model, optimizer, -1, best_minade, date_record)

    # test_main(val_loader)


def main():
    seed_everything(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = arg_parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    train(config_obj)


if __name__ == '__main__':
    main()