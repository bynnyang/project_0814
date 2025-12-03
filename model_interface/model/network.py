from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import cat

from model_interface.model.attention import AttentionNetwork

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Network(nn.Module):
    def __init__(self, layers: list, orthogonal_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(layers))
        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

    def forward(self, x):
        out = self.net(x)
        return out

class MultiObsEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        embed_size = configs['embed_size']  #128
        hidden_size = configs['hidden_size'] #256
        activate_func = [nn.LeakyReLU(), nn.Tanh()][configs['use_tanh_activate']]
        self.use_img = False if configs['img_shape'] is None else True
        self.use_action_mask = False if configs['action_mask_shape'] is None else True
        self.use_attention = False if configs['attention_configs'] is None else True
        self.input_action = 'input_action_dim' in configs and configs['input_action_dim'] > 0

        if not self.use_attention:
            if configs['n_hidden_layers'] == 1:
                layers = [nn.Linear(configs['n_modal']*embed_size, configs['output_size'])]
            else:
                layers = [nn.Linear(configs['n_modal']*embed_size, hidden_size)]
                for _ in range(configs['n_hidden_layers']-2):
                    layers.append(activate_func)
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Linear(hidden_size, configs['output_size']))
            self.net = nn.Sequential(*layers)
        else:
            attention_configs = configs['attention_configs']
            self.net = AttentionNetwork(
                embed_size,
                attention_configs['depth'],
                attention_configs['heads'],
                attention_configs['dim_head'],
                attention_configs['mlp_dim'],
                configs['n_modal'],
                attention_configs['hidden_dim'],
                configs['output_size'],
            )
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None

        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)

        if configs['target_shape'] is not None:
            layers = [nn.Linear(configs['target_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_tgt = nn.Sequential(*layers)
            
        if configs['action_mask_shape'] is not None:
            layers = [nn.Linear(configs['action_mask_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_am = nn.Sequential(*layers)

        if configs['img_shape'] is not None:
            self.embed_img = ImgEncoderStriped(configs['img_shape'], configs['k_img_conv'],\
                                    embed_size, configs['img_conv_layers'], configs['img_linear_layers'])
            self.re_embed_img = nn.Sequential(activate_func, nn.Linear(embed_size, embed_size)) # the latten vector may not be scaled

        if self.input_action:
            layers = [nn.Linear(configs['input_action_dim'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_action = nn.Sequential(*layers)

        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        def init_linear_seq(seq: nn.Sequential, last_gain: float = 0.01):
            # 找出所有 Linear 层
            linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            if not linears:
                return
            for i, m in enumerate(linears):
                gain = 1.0 if i < len(linears) - 1 else last_gain
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # 1) 主干网络 self.net
        # 如果是 MLP（非 attention），可以把最后一层 gain 设小一点
        if not self.use_attention and isinstance(self.net, nn.Sequential):
            init_linear_seq(self.net, last_gain=0.01)
        else:
            # attention 的内部结构比较复杂，这里就统一 gain=1
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # 2) lidar
        if hasattr(self, "embed_lidar"):
            for m in self.embed_lidar.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # 3) target
        if hasattr(self, "embed_tgt"):
            for m in self.embed_tgt.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # 4) action mask
        if self.use_action_mask and hasattr(self, "embed_am"):
            for m in self.embed_am.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # 5) re_embed_img（不要动预训练 encoder）
        if self.use_img and hasattr(self, "re_embed_img"):
            for m in self.re_embed_img.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # 6) input_action
        if self.input_action and hasattr(self, "embed_action"):
            for m in self.embed_action.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def load_img_encoder(self, path, device, require_grad = False):
        device = torch.device(device)

        # 1. 加载 checkpoint，直接 map 到最终 device
        ckpt = torch.load(path, map_location=device)

        # 2. 新建 AE
        ae = AE_ConvStriped(img_shape=self.cfg['img_shape'],
                     k=self.cfg['k_img_conv'],
                     embed_size=self.cfg['embed_size'],
                     c_conv_list=self.cfg['img_conv_layers'],
                     size_fc_list=self.cfg['img_linear_layers'],
                     use_tanh=False)

        # 3. 加载参数
        ae.load_state_dict(ckpt["state_dict"])

        # 4. 移到目标设备
        ae.to(device)

        # 5. 冻结/不冻
        if not require_grad:
            for p in ae.parameters():
                p.requires_grad = False

        # 6. 抽取 encoder
        self.embed_img = ae.encoder

    def forward(self, x:dict):
        '''
            x: dictionary of different input modal. Includes:

            `image` : image with shape (n, c, w, h)
            `target` : tensor in shape (n, t)
            `lidar` : tensor in shape (n, l)

        '''
        feature_lidar = self.embed_lidar(x['lidar'])
        feature_target = self.embed_tgt(x['target'])
        features = [feature_lidar, feature_target]
        if self.use_action_mask:
            feature_am = self.embed_am(x['action_mask'])
            features.append(feature_am)

        if self.use_img:
            feature_img, _ = self.embed_img(x['image'])
            feature_img = self.re_embed_img(feature_img)
            features.append(feature_img)

        if self.input_action:
            feature_action = self.embed_action(x['action'])
            features.append(feature_action)

        if self.use_attention:
            embed = torch.stack(features, dim=1)
        else:
            embed = cat(features, dim=1)
        out = self.net(embed)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, Cin, Cout, K, Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.MaxPool2d(Pooling),
            )
        # self.downSample = nn.AvgPool2d(2)
        self.shortcut = nn.Sequential(
                     nn.Conv2d(Cin, Cout, kernel_size=1),
                     nn.AvgPool2d(2)
                )
        self.res = Res
        self.cin = Cin
        self.cout = Cout
    
    def forward(self,x):
        x1 = self.layer(x)
        if self.res:
            x_res = self.shortcut(x)
            x1 = x1 + x_res
        return x1
    
class DeConvBlock(nn.Module):
    def __init__(self, Cin, Cout, K, upsample, padding=None, Batch_norm=False, Res = True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        if not padding:
            P = K//2
        else:
            P = padding
        if Batch_norm:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(Cin),
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=K,padding=P),
                activate_func,
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=K,padding=P),
            )

        self.res = Res
        self.cin = Cin
        self.cout = Cout
        self.short_cut = nn.Sequential(
                nn.ConvTranspose2d(Cin,Cout,kernel_size=1),
                nn.UpsamplingBilinear2d(upsample),
                nn.Conv2d(Cout,Cout,kernel_size=1),
            )

    def forward(self,x):
        x1 = self.layer(x)
        _, _, W, H = x.shape
        if W != H:
            raise NotImplementedError
        if self.res:
            x_res = self.short_cut(x)
            x1 = x1 + x_res
        return x1

class ImgEncoder(nn.Module):
    def __init__(self, input_shape, K, embed_size, c_conv_list, size_fc_list,\
                Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]
        Cin, w, h = input_shape
        layers = [ConvBlock(Cin, c_conv_list[0], K, Pooling, padding, Batch_norm, Res, use_tanh)]
        for i in range(len(c_conv_list)-1):
            layers.append(ConvBlock(c_conv_list[i], c_conv_list[i+1], K, Pooling, padding, Batch_norm, Res, use_tanh))
        layers.append(nn.Flatten())
        linear_input_shape = int(w*h*c_conv_list[-1]/(4**(len(c_conv_list))))
        layers.extend([nn.Linear(linear_input_shape, size_fc_list[0]), activate_func])
        for i in range(len(size_fc_list)-1):
            layers.append(nn.Linear(size_fc_list[i], size_fc_list[i+1]))
            layers.append(activate_func)
        self.net = nn.Sequential(*layers)
        self.output_mean = nn.Linear(size_fc_list[-1],embed_size)
        self.output_std = nn.Linear(size_fc_list[-1],embed_size)
    
    def forward(self, x):
        x = self.net(x)
        return self.output_mean(x), self.output_std(x)
    
class ImgDecoder(nn.Module):
    def __init__(self, output_shape, K, embed_size, c_conv_list, size_fc_list,\
                 padding=None, Batch_norm=False, Res=True, use_tanh=True):
        super().__init__()
        out_channel, w, h = output_shape
        self.output_shape = output_shape
        self.c_conv_list = c_conv_list
        activate_func = [nn.LeakyReLU(), nn.Tanh()][use_tanh]

        # The fully connected layers
        fc_list = []
        fc_list.extend([nn.Linear(embed_size, size_fc_list[-1]), activate_func])
        for i in range(len(size_fc_list)-1):
            fc_list.append(nn.Linear(size_fc_list[-(i+1)], size_fc_list[-(i+2)]))
            fc_list.append(activate_func)
        fc_output_size = int( h/(2**len(c_conv_list)) * w/(2**len(c_conv_list)) * c_conv_list[-1] )
        fc_list.extend([nn.Linear(size_fc_list[0], fc_output_size), activate_func])
        self.fc_net = nn.Sequential(*fc_list)

        # The DeConvolution layers
        conv_list = []
        upsample_size = int(h/(2**len(c_conv_list))*2)
        for i in range(len(c_conv_list)-1):
            conv_list.append(DeConvBlock(c_conv_list[-i-1],c_conv_list[-i-2],K,upsample_size, padding, Batch_norm, Res, use_tanh))
            upsample_size *= 2
        conv_list.append(DeConvBlock(c_conv_list[0],out_channel,K,upsample_size, padding, Batch_norm, Res, use_tanh))
        self.conv_net = nn.Sequential(*conv_list)

        self.output = nn.Sigmoid()
    
    def forward(self, z):
        x = self.fc_net(z)
        B = x.shape[0]
        # assume the image: w==h
        w = int(self.output_shape[-2]/(2**len(self.c_conv_list)))
        x = x.reshape((B, self.c_conv_list[-1], w, w))
        x = self.conv_net(x)
        x = self.output(x)
        return x
    
class VAE_Conv(nn.Module):
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        super(VAE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        sampled_z = self.sampling(mean,std)
        return self.decoder(sampled_z), mean, std
    
    def eval_forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,std = self.encoder(x)
        # sampled_z = self.sampling(mean,std)
        return self.decoder(mean), mean, std

    def embed(self,img):
        '''
        Input:
            img: torch.Tensor of shape (B, C, W, H)
        Return:
            embed_img: torch.Tensor of Shape (B, HIDDEN_SIZE)
        '''
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img,_ = self.encoder(img)
        return embed_img
    
    def save(self, path):
        torch.save(self, path)
        print('save model in: %s'%path)

class AE_Conv(nn.Module):
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        super(AE_Conv, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoder(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,_ = self.encoder(x)
        return self.decoder(mean)

    def embed(self,img):
        '''
        Input:
            img: torch.Tensor of shape (B, C, W, H)
        Return:
            embed_img: torch.Tensor of Shape (B, HIDDEN_SIZE)
        '''
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img, embed_std = self.encoder(img)
        return embed_img, embed_std

    def save(self, path):
        torch.save(self, path)
        print('save model in: %s'%path)


class StripedMultiHeadMLP(nn.Module):
    def __init__(self, input_channels, spatial_size, channel_heads, strip_num,
                 fc_dims, embed_size, use_tanh=True, shared_strips=True):
        """
        Args:
            input_channels: 输入通道数 (8)
            spatial_size: 空间尺寸 (128, 128)
            channel_heads: 通道分组数 (8)
            strip_num: 每个通道的空间条带数 (2)
            fc_dims: MLP隐藏层维度列表, e.g. [256, ]
            embed_size: 最终输出嵌入维度
            shared_strips: 是否所有通道共享同一组条带MLP（推荐True，大幅减参）
        """
        super().__init__()
        self.channel_heads = channel_heads  # 8
        self.strip_num = strip_num  # 2
        self.channels_per_head = input_channels // channel_heads  # 1
        self.use_tanh = use_tanh
        self.shared_strips = shared_strips
        
        assert input_channels % channel_heads == 0, "通道数必须能被头数整除"
        assert spatial_size[1] % strip_num == 0, "宽度必须能被条带数整除"
        
        self.strip_h = spatial_size[0]  # 128
        self.strip_w = spatial_size[1] // strip_num  # 64
        
        self.activate_func = nn.Tanh() if use_tanh else nn.LeakyReLU()

        strip_input_dim = self.channels_per_head * self.strip_h * self.strip_w
        
         # 重构MLP存储方式：一维列表，便于索引
        self._build_mlps(strip_input_dim, fc_dims)
        
        # 最终输出层
        total_mlp_output = fc_dims[-1] * strip_num * channel_heads  # 256*2*8=4096
        self.output_mean = nn.Linear(total_mlp_output, embed_size)
        self.output_std = nn.Linear(total_mlp_output, embed_size)

    def _build_mlps(self, strip_input_dim, fc_dims):
        """构建MLP模块"""
        self.mlp_list = nn.ModuleList()
        mlp_count = self.strip_num if self.shared_strips else self.channel_heads * self.strip_num
        
        for _ in range(mlp_count):
            layers = []
            prev_dim = strip_input_dim
            for dim in fc_dims:
                layers.extend([nn.Linear(prev_dim, dim), self.activate_func])
                prev_dim = dim
            self.mlp_list.append(nn.Sequential(*layers))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        strip_outputs = []
        
        # 遍历每个头：通过计算通道索引，避免5维张量
        for h in range(self.channel_heads):
            # 计算当前头在通道维度的起始和结束位置
            start_channel = h * self.channels_per_head
            end_channel = start_channel + self.channels_per_head
            
            # 直接切片提取头的特征图：(B, channels_per_head, H, W) -> 4维！
            head_x = x[:, start_channel:end_channel, :, :]
            
            # 遍历每个条带
            for s in range(self.strip_num):
                start_w = s * self.strip_w
                end_w = start_w + self.strip_w
                
                # 提取条带：(B, channels_per_head, H, strip_w) -> 4维！
                strip = head_x[:, :, :, start_w:end_w]
                
                # 展平：使用reshape，安全
                strip_flat = strip.reshape(B, -1)
                
                # 选择正确的MLP
                if self.shared_strips:
                    mlp_idx = s  # 共享模式：只根据条带索引
                else:
                    mlp_idx = h * self.strip_num + s  # 独立模式：组合索引
                
                strip_out = self.mlp_list[mlp_idx](strip_flat)
                strip_outputs.append(strip_out)
        
        # 拼接所有条带输出: (B, fc_dims[-1]*16)
        x = torch.cat(strip_outputs, dim=-1)
        
        return self.output_mean(x), self.output_std(x)
    

# 修改后的ImgEncoder
class ImgEncoderStriped(nn.Module):
    def __init__(self, input_shape, K, embed_size, c_conv_list, size_fc_list,
                 Pooling=2, padding=None, Batch_norm=False, Res=True, use_tanh=True,
                 channel_heads=8, strip_num=2, shared_strips = False):
        super().__init__()
        activate_func = nn.Tanh() if use_tanh else nn.LeakyReLU()
        Cin, w, h = input_shape
        
        # 卷积部分不变
        layers = [ConvBlock(Cin, c_conv_list[0], K, Pooling, padding, Batch_norm, Res, use_tanh)]
        for i in range(len(c_conv_list)-1):
            layers.append(ConvBlock(c_conv_list[i], c_conv_list[i+1], K, Pooling, padding, Batch_norm, Res, use_tanh))
        self.conv_net = nn.Sequential(*layers)
        
        # 计算最终特征图大小
        num_pooling = len(c_conv_list)
        final_h, final_w = h // (2**num_pooling), w // (2**num_pooling)
        
        # 条带分割MLP
        self.striped_mlp = StripedMultiHeadMLP(
            input_channels=c_conv_list[-1],
            spatial_size=(final_h, final_w),
            channel_heads=channel_heads,  # 8
            strip_num=strip_num,          # 2
            fc_dims=size_fc_list,         # 如 [128, 64]
            embed_size=embed_size,
            use_tanh=use_tanh,
            shared_strips= shared_strips  # 强烈推荐，参数量降低8倍
        )
    
    def forward(self, x):
        x = self.conv_net(x)  # (B, 8, 128, 128)
        return self.striped_mlp(x)
    

class StripedMultiHeadMLPDecoder(nn.Module):
    def __init__(self, input_channels, spatial_size, channel_heads, strip_num,
                 fc_dims, embed_size, use_tanh=True, shared_strips=True):
        """
        条带分割MLP解码器（编码器的镜像结构）
        Args:
            input_channels: 解码器输出的通道数（与编码器输入通道一致）
            spatial_size: 目标空间尺寸 (128, 128)
            channel_heads: 通道分组数（需与编码器一致）
            strip_num: 每个通道的空间条带数（需与编码器一致）
            fc_dims: 编码器MLP的隐藏层维度列表（将自动反转）
            embed_size: 输入嵌入向量的维度
            shared_strips: 是否共享条带MLP参数（需与编码器一致）
        """
        super().__init__()
        self.channel_heads = channel_heads
        self.strip_num = strip_num
        self.channels_per_head = input_channels // channel_heads
        self.shared_strips = shared_strips
        
        # 计算条带维度
        self.strip_h = spatial_size[0]
        self.strip_w = spatial_size[1] // strip_num
        self.strip_output_dim = self.channels_per_head * self.strip_h * self.strip_w  # 8192
        
        # 激活函数
        self.activate_func = nn.Tanh() if use_tanh else nn.LeakyReLU()
        self.strip_input_dim = fc_dims[-1]  # 编码器MLP的输出维度（解码器的输入维度）
        
        # 构建解码MLP（结构是编码器的镜像）
        self._build_decoder_mlps(fc_dims)
        
        # 输入投影层：将嵌入向量映射到条带输入空间
        total_mlp_input = self.strip_input_dim * channel_heads * strip_num
        self.input_projection = nn.Linear(embed_size, total_mlp_input)
    
    def _build_decoder_mlps(self, fc_dims):
        """构建解码MLP：从低维嵌入重建高维特徵"""
        # 编码器：8192 -> [hidden_dims] -> 256
        # 解码器：256 -> [hidden_dims逆序] -> 8192

        mlp_count = self.strip_num if self.shared_strips else self.channel_heads * self.strip_num
        
        # 一维列表存储所有MLP，便于索引
        self.mlp_list = nn.ModuleList()
        
        for _ in range(mlp_count):
            layers = []
            prev_dim = self.strip_input_dim
            
            # 如果fc_dims有多个层，需要逆序构建
            # 例如 fc_dims = [256, 128] -> 解码器为 256 -> 128 -> 8192
            for dim in reversed(fc_dims[:-1]):
                layers.extend([nn.Linear(prev_dim, dim), self.activate_func])
                prev_dim = dim
            
            # 最后一层映射到条带输出维度
            layers.extend([nn.Linear(prev_dim, self.strip_output_dim), self.activate_func])
            
            self.mlp_list.append(nn.Sequential(*layers))
    
    # def _create_mlp_decoder(self, fc_dims):
    #     """创建单个解码MLP（编码器MLP的镜像）"""
    #     layers = []
    #     prev_dim = self.strip_input_dim
        
    #     # 反转隐藏层维度（如果有多个层）
    #     # 例如编码器是 [256, 128] -> 64，解码器是 64 -> [128, 256] -> 8192
    #     for dim in reversed(fc_dims[:-1]):
    #         layers.extend([nn.Linear(prev_dim, dim), self.activate_func])
    #         prev_dim = dim
        
    #     # 最后一层映射到条带输出维度
    #     layers.extend([nn.Linear(prev_dim, self.strip_output_dim), self.activate_func])
        
    #     return nn.Sequential(*layers)
    
    def forward(self, z):
        """
        前向传播
        Args:
            z: 嵌入向量 (B, embed_size)
        Returns:
            重建的特征图 (B, input_channels, H, W)
        """
        B = z.size(0)
        
        # Step 1: 投影到条带输入空间
        # (B, embed_size) -> (B, total_mlp_input)
        x = self.input_projection(z)
        
        # Step 2: 分割成多个条带输入
        # 16个 (B, strip_input_dim)，每个256维
        strip_inputs = x.chunk(self.channel_heads * self.strip_num, dim=-1)
        
        # Step 3: 逐个条带解码并重组
        head_outputs = []
        
        for h in range(self.channel_heads):
            head_strips = []
            for s in range(self.strip_num):
                # 当前条带的输入 (B, 256)
                input_idx = h * self.strip_num + s
                strip_input = strip_inputs[input_idx]
                
                # 解码回高维 (B, 8192)
                if self.shared_strips:
                    mlp_idx = s
                else:
                    mlp_idx = input_idx

                strip_output = self.mlp_list[mlp_idx](strip_input)
                
                # 重塑为空间结构 (B, channels_per_head, strip_h, strip_w)
                strip_output = strip_output.reshape(B, self.channels_per_head, self.strip_h, self.strip_w)
                head_strips.append(strip_output)
                
            
            # 在宽度维度拼接当前头的所有条带
            # 2个 (B, 1, 128, 64) -> (B, 1, 128, 128)
            head_output = torch.cat(head_strips, dim=-1)
            head_outputs.append(head_output)
        
        # Step 4: 在通道维度拼接所有头
        # 8个 (B, 1, 128, 128) -> (B, 8, 128, 128)
        return torch.cat(head_outputs, dim=1)


class ImgDecoderStriped(nn.Module):
    def __init__(self, output_shape, K, embed_size, c_conv_list, size_fc_list,
                 padding=None, Batch_norm=False, Res=True, use_tanh=True,
                 channel_heads=8, strip_num=2, shared_strips=False):
        """
        条带分割解码器
        Args:
            output_shape: 输出图像形状 (C, H, W)
            K: 卷积核大小
            embed_size: 嵌入向量维度（与编码器输出一致）
            c_conv_list: 卷积层通道列表（编码器的逆序）
            size_fc_list: MLP隐藏层维度（与编码器一致）
            channel_heads: 通道分组数（需与编码器一致）
            strip_num: 条带数（需与编码器一致）
            shared_strips: 是否共享MLP（需与编码器一致）
        """
        super().__init__()
        out_channel, w, h = output_shape
        self.output_shape = output_shape
        self.c_conv_list = c_conv_list
        
        # 计算编码器输出的空间尺寸
        num_pooling = len(c_conv_list)
        final_h, final_w = h // (2**num_pooling), w // (2**num_pooling)
        
        # 1. 条带解码MLP（重建特征图）
        self.striped_mlp_decoder = StripedMultiHeadMLPDecoder(
            input_channels=c_conv_list[-1],
            spatial_size=(final_h, final_w),
            channel_heads=channel_heads,
            strip_num=strip_num,
            fc_dims=size_fc_list,
            embed_size=embed_size,
            use_tanh=use_tanh,
            shared_strips=shared_strips
        )
        
        # 2. 反卷积网络（上采样）
        self.conv_net = self._build_deconv_net(K, padding, Batch_norm, Res, use_tanh)
        
        # 3. 输出激活函数
        self.output = nn.Sigmoid()
    
    def _build_deconv_net(self, K, padding, Batch_norm, Res, use_tanh):
        """构建反卷积网络"""
        conv_list = []
        upsample_size = int(self.output_shape[-1] // (2 ** len(self.c_conv_list)) * 2)
        
        # 反向遍历通道列表 (c_conv_list[-1] -> c_conv_list[0])
        for i in range(len(self.c_conv_list) - 1):
            in_channels = self.c_conv_list[-(i + 1)]
            out_channels = self.c_conv_list[-(i + 2)]
            conv_list.append(
                DeConvBlock(in_channels, out_channels, K, upsample_size, 
                           padding, Batch_norm, Res, use_tanh)
            )
            upsample_size *= 2
        
        # 最后一层到输出通道
        conv_list.append(
            DeConvBlock(self.c_conv_list[0], self.output_shape[0], K, upsample_size,
                       padding, Batch_norm, Res, use_tanh)
        )
        
        return nn.Sequential(*conv_list)
    
    def forward(self, z):
        """
        前向传播
        Args:
            z: 嵌入向量 (B, embed_size)
        Returns:
            重建图像 (B, C, H, W)
        """
        # Step 1: 通过条带MLP解码为特征图
        # (B, embed_size) -> (B, c_conv_list[-1], H//2^L, W//2^L)
        x = self.striped_mlp_decoder(z)
        
        # Step 2: 通过反卷积上采样
        x = self.conv_net(x)
        
        # Step 3: 输出激活
        return self.output(x)
    

class AE_ConvStriped(nn.Module):
    def __init__(self, img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=False):
        super(AE_ConvStriped, self).__init__()
        self.z_size = embed_size
        self.encoder = ImgEncoderStriped(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        self.decoder = ImgDecoderStriped(img_shape, k, embed_size, c_conv_list, size_fc_list, use_tanh=use_tanh)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        mean,_ = self.encoder(x)
        return self.decoder(mean)

    def embed(self,img):
        '''
        Input:
            img: torch.Tensor of shape (B, C, W, H)
        Return:
            embed_img: torch.Tensor of Shape (B, HIDDEN_SIZE)
        '''
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            embed_img, embed_std = self.encoder(img)
        return embed_img, embed_std

    def save(self, path):
        torch.save(self, path)
        print('save model in: %s'%path)