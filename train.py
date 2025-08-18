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
from utils.eval import get_eval_metric_results
from model_interface.model.parking_model_real import ParkingModelReal
from torch_geometric.data import Batch
from torch.utils.data._utils.collate import default_collate


decay_lr_factor = 0.3
decay_lr_every = 5
lr = 0.001
epochs = 25
end_epoch = 0
lr = 0.001
in_channels, out_channels = 8, 60
show_every = 20
val_every = 5
best_minade = float('inf')
save_dir = './trained_params'
date_record = "250815"
global_step = 0


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
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{end_epoch}.valminade_{val_minade:.3f}.{date}.{"xkhuang"}.pth')
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
    tensor_keys = ['gt_traj_point', 'gt_traj_point_token',
                   'target_point', 'fuzzy_target_point']
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
   
    dataset = ParkingDataModuleReal(config_obj, is_train=1)
    train_loader = DataLoader(dataset, batch_size=config_obj.batch_size, shuffle=True, num_workers=config_obj.num_workers, collate_fn=collate_graph)
    val_loader = DataLoader(dataset, batch_size= config_obj.batch_size, shuffle=False, num_workers=config_obj.num_workers, collate_fn=collate_graph)

    model = ParkingModelReal(config_obj)
    model = model.to(device=device)
    traj_point_loss_func = TokenTrajPointLoss(config_obj)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    


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
            if epoch < end_epoch: break
            optimizer.zero_grad()
            out = model(data)
            loss = traj_point_loss_func(out, data)
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
            metrics = get_eval_metric_results(config_obj, model, val_loader, device)
            curr_minade = metrics
            print(f"minADE:{metrics:3f}")

            if curr_minade < best_minade:
                best_minade = curr_minade
                save_checkpoint(save_dir, model, optimizer, epoch, best_minade, date_record)
                
    # eval result on the identity dataset
    metrics = get_eval_metric_results(config_obj, model, val_loader, device)
    curr_minade = metrics
    if curr_minade < best_minade:
        best_minade = curr_minade
        save_checkpoint(save_dir, model, optimizer, -1, best_minade, date_record)


def main():
    seed_everything(16)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='./config/training_real.yaml', type=str)
    args = arg_parser.parse_args()
    config_path = args.config
    config_obj = get_train_config_obj(config_path)

    train(config_obj)


if __name__ == '__main__':
    main()