import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_interface.dataloader import ParkingDataloaderModule
from dataset_interface.dataset_real import ParkingDataModuleReal
from utils.config import get_train_config_obj
from dataset import GraphData
from dataset import GraphDataset
from torch.utils.data import DataLoader



def train(config_obj):
   
    dataset = ParkingDataModuleReal(config_obj, is_train=1)
    train_loader = DataLoader(dataset, batch_size= config_obj.batch_size, shuffle=True, num_workers=config_obj.num_workers)
    epochs = 25
    for epoch in range(epochs):
        print(epoch)
        for data in train_loader:
            hid = 1 


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