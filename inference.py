import argparse

from pytorch_lightning.loggers import TensorBoardLogger

from dataset_interface.dataloader import ParkingDataloaderModule
from dataset_interface.dataset_real import ParkingDataModuleReal
from utils.config import get_inference_config_obj
from dataset import GraphData
from dataset import GraphDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import time
from loss.traj_point_loss import TokenTrajPointLoss
from torch_geometric.data import Batch
from torch.utils.data._utils.collate import default_collate
from utils.config import InferenceConfiguration
from model_interface.model_interface import get_parking_model
from tqdm import tqdm

    
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
    # tensor_keys = ['gt_traj_point', 'gt_traj_point_token']
    tensor_keys = ['gt_traj_point_token', 'target_point']
    tensor_dict = default_collate([{k: getattr(g, k) for k in tensor_keys}
                                   for g in batch_list])

    # 把张量挂到大图上
    for k, v in tensor_dict.items():
        setattr(big_graph, k, v)

    return big_graph

def inference(inference_cfg: InferenceConfiguration):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg)
    dataset_test = ParkingDataModuleReal(inference_cfg.train_meta_config, is_train=2)
    test_loader = DataLoader(dataset_test, batch_size= 1, shuffle=False, num_workers=0, collate_fn=collate_graph)
    count = 0
    for data in tqdm(test_loader):
        data.to(device)
        parking_inference_obj.predict(data, count, mode=inference_cfg.predict_mode)
        count+=1
    



def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inference_config_path', default='./config/inference_real.yaml', type=str)
    args = arg_parser.parse_args()
    inference_cfg = get_inference_config_obj(args.inference_config_path)

    inference(inference_cfg)

def test_main(test_loader):
    inference_cfg = get_inference_config_obj('./config/inference_real.yaml')

    inference(inference_cfg, test_loader)


if __name__ == '__main__':
    main()