import torch
from torch import nn

class SmootherResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class Smoother(nn.Module):
    def __init__(self, window_size, output_size, 
                 hidden_size=512, res_hidden_size=256, 
                 num_blocks=3, dropout=0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks= num_blocks
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True)
        )

        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmootherResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        N, C, T = x.shape

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x
    

class MotionSmoother(nn.Module):
    def __init__(self, window_size, output_size, 
                 hidden_size=512, res_hidden_size=256, 
                 num_blocks=3, dropout=0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks= num_blocks
        self.dropout = dropout

        self.pos_smoother = Smoother(
                window_size,
                output_size,
                hidden_size,
                res_hidden_size,
                num_blocks,
                dropout,
            )
        self.vel_smoother = Smoother(
                window_size-1,
                output_size,
                hidden_size,
                res_hidden_size,
                num_blocks,
                dropout,
            )
        self.acc_smoother = Smoother(
                window_size-2,
                output_size,
                hidden_size,
                res_hidden_size,
                num_blocks,
                dropout,
            )
        
        self.fusion_layer = nn.Linear(
            3*output_size, output_size
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N, C, T = x.shape

        assert T == self.window_size, (
            'Input sequence length must be equal to the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Forward layers
        pos = x
        vel = x[..., 1:]-x[..., :-1]
        acc = vel[..., 1:]-vel[..., :-1]
        x_pos = self.pos_smoother(pos)
        x_vel = self.vel_smoother(vel)
        x_acc = self.acc_smoother(acc)
        x = torch.cat([x_pos, x_vel, x_acc], dim=2)
        x = self.fusion_layer(x)
        x = x.permute(0, 2, 1)
        return x