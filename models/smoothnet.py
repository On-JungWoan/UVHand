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


class ArcticSmoother(nn.Module):
    def __init__(self, batch_size, window_size):
        super().__init__()

        self.hand_v_smoother = MotionSmoother(window_size, window_size)
        self.obj_v_smoother = MotionSmoother(window_size, window_size)

        self.batch_size = batch_size
        self.window_size = window_size

    def forward(self, pred_vl, pred_vr, pred_vo):
        B = self.batch_size
        T = self.window_size

        sm_l_v = self.hand_v_smoother(pred_vl.view(B, T, -1)).reshape(B*T, -1, 3)
        sm_r_v = self.hand_v_smoother(pred_vr.view(B, T, -1)).reshape(B*T, -1, 3)
        sm_o_v = self.obj_v_smoother(pred_vo.view(B, T, -1)).reshape(B*T, -1, 3)

        return sm_l_v, sm_r_v, sm_o_v


class OldArcticSmoother(nn.Module):
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
    def __init__(self, batch_size, window_size, weight_dict):
        super().__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.weight_dict = weight_dict
        self.losses = ['smooth']

    def get_loss(self, loss, arctic_pred, arctic_gt, meta_info, args):
        loss_map = {
            'smooth': self.loss_smooth,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](arctic_pred, arctic_gt, meta_info, args)
    
    def pre_process_vertex(self, pred, targets):
        gt_vo = targets["object.v.cam"]
        gt_vr = targets["mano.v3d.cam.r"]
        gt_vl = targets["mano.v3d.cam.l"]

        pred_vo = pred["object.v.cam"]
        pred_vr = pred["mano.v3d.cam.r"]
        pred_vl = pred["mano.v3d.cam.l"]

        # hand roots
        pred_root_r = pred["mano.j3d.cam.r"][:, :1]
        pred_root_l = pred["mano.j3d.cam.l"][:, :1]
        gt_root_r = targets["mano.j3d.cam.r"][:, :1]
        gt_root_l = targets["mano.j3d.cam.l"][:, :1]

        # object roots
        parts_ids = targets["object.parts_ids"]
        bottom_idx = parts_ids[0] == 2
        gt_root_o = gt_vo[:, bottom_idx].mean(dim=1)[:, None, :]
        pred_root_o = pred_vo[:, bottom_idx].mean(dim=1)[:, None, :]

        # root relative (num_frames, num_verts, 3)
        gt_vr_ra = gt_vr - gt_root_r
        gt_vl_ra = gt_vl - gt_root_l
        gt_vo_ra = gt_vo - gt_root_o

        # root relative (num_frames, num_verts, 3)
        pred_vr_ra = pred_vr - pred_root_r
        pred_vl_ra = pred_vl - pred_root_l
        pred_vo_ra = pred_vo - pred_root_o

        is_valid = targets["is_valid"]
        left_valid = targets["left_valid"] * is_valid
        right_valid = targets["right_valid"] * is_valid

        return gt_vr_ra, gt_vl_ra, gt_vo_ra, \
            pred_vr_ra, pred_vl_ra, pred_vo_ra, \
            is_valid, left_valid, right_valid
    
    def compute_acc_vel_loss(self, pred, gt, criterion, valid):
        _, V, D = pred.shape
        valid = valid.unsqueeze(-1).unsqueeze(-1).repeat(1,V,D)
        pred = pred * valid
        gt = gt * valid

        pred = pred.reshape(self.batch_size, self.window_size, -1)
        gt = gt.reshape(self.batch_size, self.window_size, -1)

        pred = pred.permute(0,2,1) # N, C, T
        gt = gt.permute(0,2,1) # N, C, T

        # velocity
        vel_pred = pred[..., 1:] - pred[..., :-1]
        vel_gt = gt[..., 1:] - gt[..., :-1]

        # accel
        acc_pred = vel_pred[..., 1:] - vel_pred[..., :-1]
        acc_gt = vel_gt[..., 1:] - vel_gt[..., :-1]

        loss_vel = criterion(vel_pred, vel_gt).mean()
        loss_acc = criterion(acc_pred, acc_gt).mean()

        return 1.0 * loss_vel + 1.0 * loss_acc

    def loss_smooth(self, arctic_pred, arctic_gt, meta_info, args):
        mse_loss = nn.MSELoss(reduction="none")

        # not root aligned
        gt_vo = arctic_gt["object.v.cam"]
        gt_vr = arctic_gt["mano.v3d.cam.r"]
        gt_vl = arctic_gt["mano.v3d.cam.l"]
        pred_vo = arctic_pred["object.v.cam"]
        pred_vr = arctic_pred["mano.v3d.cam.r"]
        pred_vl = arctic_pred["mano.v3d.cam.l"]

        # root aligned
        gt_vr_ra, gt_vl_ra, gt_vo_ra, \
        pred_vr_ra, pred_vl_ra, pred_vo_ra, \
        is_valid, left_valid, right_valid = self.pre_process_vertex(arctic_pred, arctic_gt)

        loss_right = self.compute_acc_vel_loss(pred_vr, gt_vr, mse_loss, right_valid)
        loss_left = self.compute_acc_vel_loss(pred_vl, gt_vl, mse_loss, left_valid)
        loss_obj = self.compute_acc_vel_loss(pred_vo, gt_vo, mse_loss, is_valid)        

        losses = {}
        losses['loss_left'] = loss_left
        losses['loss_right'] = loss_right
        losses['loss_obj'] = loss_obj
        return losses

    def forward(self, args, data, meta_info):
        arctic_pred = data.search('pred.', replace_to='')
        arctic_gt = data.search('targets.', replace_to='')

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, arctic_pred, arctic_gt, meta_info, args))
        # losses.update(compute_loss(arctic_pred, arctic_gt, meta_info, args))

        return losses