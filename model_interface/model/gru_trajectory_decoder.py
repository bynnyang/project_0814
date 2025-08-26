import torch
from torch import nn

from utils.config import Configuration


class GRUTrajectoryDecoder(nn.Module):
    def __init__(self, cfg: Configuration):
        super(GRUTrajectoryDecoder, self).__init__()

        self.cfg = cfg
        self.predict_num = self.cfg.autoregressive_points
        self.hidden_size = 1024
        self.num_layers = 5


        self.join = nn.Sequential(
            nn.Linear(512, 256),
            # nn.LayerNorm(256),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(256, self.hidden_size)
        )

        self.predict_item_number = self.cfg.item_number
        
        

        self.decoder = nn.GRU(input_size=self.predict_item_number, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.00001)

    

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, cfg.item_number),
            nn.Tanh()
        )
        
        self.init_weights()


    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, data_gt=None, teacher_forcing_ratio=0.5):
        z = torch.flatten(z, 1)
        h0 = self.join(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # 2层GRU

        use_teacher_forcing = False if (self.training and data_gt is not None and torch.rand(1).item() < teacher_forcing_ratio) else False
        
        if data_gt != None:
            y_gt = data_gt.reshape(data_gt.size(0), self.predict_num, self.cfg.item_number)
        outputs = []
        
        if use_teacher_forcing and y_gt.size(1) < self.predict_num:
            raise ValueError(f"y_gt sequence length ({y_gt.size(1)}) is less than predict_num ({self.predict_num})")
        
        step_in = torch.zeros(z.size(0), 1, self.cfg.item_number, device=z.device)

        for t in range(self.predict_num):
            out, h0 = self.decoder(step_in, h0)
            pred = self.out(out.squeeze(1))
            current_output = pred + step_in.squeeze(1)  # 残差连接
            outputs.append(current_output.unsqueeze(1))
            if t < self.predict_num - 1:  # 不需要为最后一个时间步准备输入
                if use_teacher_forcing:
                    # 使用上一步的真实值作为下一步的输入
                    step_in = y_gt[:, t:t+1].clone()  # 使用当前时间步t的真实值作为下一步t+1的输入
                else:
                    # 使用上一步的预测值作为下一步的输入
                    step_in = current_output.unsqueeze(1)

        x = torch.cat(outputs, dim=1)
        return x
    