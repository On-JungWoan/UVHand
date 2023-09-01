import torch
from torch import nn
from arctic_tools.process import make_output
from arctic_tools.src.callbacks.loss.loss_arctic_sf import compute_loss

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


class ArcticSmoother(nn.Module):
    def __init__(self, batch_size, window_size):
        super().__init__()

        self.mano_pose_smoother = MotionSmoother(window_size, batch_size)
        self.mano_shape_smoother = MotionSmoother(window_size, batch_size)
        self.obj_rot_smoother = MotionSmoother(window_size, batch_size)
        self.obj_rad_smoother = MotionSmoother(window_size, batch_size)
        self.mano_root_smoother = MotionSmoother(window_size, batch_size)
        self.obj_root_smoother = MotionSmoother(window_size, batch_size)

        self.batch_size = batch_size
        self.window_size = window_size

    def forward(self, output):
        # select query
        root, mano_pose, mano_shape, obj_angle = output
        root_l, root_r, root_o = root
        mano_pose_l, mano_pose_r = mano_pose
        mano_shape_l, mano_shape_r = mano_shape
        obj_rot, obj_rad = obj_angle

        smoothed_root = [
            self.mano_root_smoother(root_l.view(-1, self.window_size, 3)).reshape(-1, 3),
            self.mano_root_smoother(root_r.view(-1, self.window_size, 3)).reshape(-1, 3),
            self.obj_root_smoother(root_o.view(-1, self.window_size, 3)).reshape(-1, 3)
        ]
        smoothed_pose = [
            self.mano_pose_smoother(mano_pose_l.view(-1, self.window_size, 48)).reshape(-1, 48),
            self.mano_pose_smoother(mano_pose_r.view(-1, self.window_size, 48)).reshape(-1, 48)
        ]
        smoothed_shape = [
            self.mano_shape_smoother(mano_shape_l.view(-1, self.window_size, 10)).reshape(-1, 10),
            self.mano_shape_smoother(mano_shape_r.view(-1, self.window_size, 10)).reshape(-1, 10)
        ]
        smoothed_obj = [
            self.obj_rot_smoother(obj_rot.view(-1, self.window_size, 3)).reshape(-1,3),
            self.obj_rad_smoother(obj_rad.view(-1, self.window_size, 1)).reshape(-1,1)
        ]

        return smoothed_root, smoothed_pose, smoothed_shape, smoothed_obj


class SmoothCriterion(nn.Module):
    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict

    def forward(self, args, data, meta_info):
        arctic_pred = data.search('pred.', replace_to='')
        arctic_gt = data.search('targets.', replace_to='')

        losses = {}
        losses.update(compute_loss(arctic_pred, arctic_gt, meta_info, args))

        return losses