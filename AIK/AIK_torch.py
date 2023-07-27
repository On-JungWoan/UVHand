import numpy as np
import torch
import AIK.AIK_RotationClass_torch as AR
import AIK.AIK_config as cfg


angels0 = np.zeros((1, 21))

def rotation_to_axis_angle(rotation_mat):
    '''
    1. This is using loop with per rotation_mat.
    
    num_j = rotation_mat.shape[1]
    # for i in range(num_j):
    #     # np_axis_angle[i*3:(i*3)+3] = R.from_matrix(np.array(rotation_mat[0][i].detach())).as_rotvec()
    #     axis_angle[i*3:(i*3)+3] = AR.CustomRotation.from_matrix(rotation_mat[0][i]).as_rotvec()
    #     #axis_angle[i*3:(i*3)+3] = AR.CustomRotation.from_matrix(rotation_mat[0][i]).as_rotvec()
        
    2. Below is Batch ver. 
    
    '''
    # axis_angle = torch.zeros(48, device=rotation_mat.device)

    # axis_angle = AR.CustomRotation.from_matrix(rotation_mat[0]).as_rotvec().reshape(-1)

    axis_angle = torch.zeros(rotation_mat.shape[0], 48).to(rotation_mat.device)

    for idx in range(rotation_mat.shape[0]):
        axis_angle[idx] = AR.CustomRotation.from_matrix(rotation_mat[idx]).as_rotvec().reshape(-1)

    return axis_angle


def to_dict_new(joints):
    '''
    No use loop version with "to_dict"
    '''

    temp_dict = dict()
    temp_dict = torch.transpose(joints,0,1).reshape(-1,3,1)

    return temp_dict

def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    Revised: SeongyeongLee
    '''
    x, y, z = axis
    if not is_normalized:
        n = torch.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
        
    c = torch.cos(torch.Tensor([angle])).to(x.device); s = torch.sin(torch.Tensor([angle])).to(x.device); C = 1-c
    # c = torch.cos(angle); s = torch.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    # return torch.tensor([
    #         [ x*xC+c,   xyC-zs,   zxC+ys ],
    #         [ xyC+zs,   y*yC+c,   yzC-xs ],
    #         [ zxC-ys,   yzC+xs,   z*zC+c ]], requires_grad=True)
    
    return torch.stack([x*xC+c,   xyC-zs,   zxC+ys,xyC+zs,   y*yC+c,   yzC-xs, zxC-ys,   yzC+xs,   z*zC+c ], dim = 0).reshape(3,3)


def adaptive_IK(T_, P_):
    '''
    Computes pose parameters given template and predictions.
    We think the twist of hand bone could be omitted.
    :param T: template, 21*3
    :param P: target, 21*3
    :return: pose params.
    '''

    gt_rotatiation = np.array([])
    
    for _, idx in enumerate(range(P_.shape[0])):
                
        T = T_.clone().to(torch.float32)
        P = P_[idx].clone().to(torch.float32)
        
        T = T.transpose(1, 0)
        P = P.transpose(1, 0)
        
        # to dict
        P = to_dict_new(P)
        T = to_dict_new(T)
        
        # some globals
        R = {}
        R_pa_k = {}
        q = {}

        q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

        # compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix.
        # you can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein"
        # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, in which R0 is regard as orthogonal matrix only.
        # Using their method might further boost accuracy.
        
        P_0 = torch.cat([P[1] - P[0], P[5] - P[0],
                            P[9] - P[0], P[13] - P[0],
                            P[17] - P[0]], axis=-1)
        
        T_0 = torch.cat([T[1] - T[0], T[5] - T[0],
                            T[9] - T[0], T[13] - T[0],
                            T[17] - T[0]], axis=-1)
        
        H = torch.matmul(T_0, P_0.T)

        U, S, V = torch.svd(H)

        R0 = torch.matmul(V, U.T)

        det0 = torch.det(R0)

        if abs(det0 + 1) < 1e-6:
            V_ = V.clone()

            if (abs(S) < 1e-4).sum():
                V_[:, 2] = -V_[:, 2]
                R0 = torch.matmul(V_, U.T)

        R[0] = R0

        # the bone from 1,5,9,13,17 to 0 has same rotations
        R[1] = R[0].clone()
        R[5] = R[0].clone()
        R[9] = R[0].clone()
        R[13] = R[0].clone()
        R[17] = R[0].clone()

        # compute rotation along kinematics
        for k in cfg.kinematic_tree:
            pa = cfg.SNAP_PARENT[k]
            pa_pa = cfg.SNAP_PARENT[pa]
            q[pa] = torch.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]
            delta_p_k = torch.matmul(torch.inverse(R[pa]), P[k] - q[pa])
            delta_p_k = delta_p_k.reshape((3,))

            delta_t_k = T[k] - T[pa]
            delta_t_k = delta_t_k.reshape((3,))
            
            temp_axis = torch.cross(delta_t_k, delta_p_k)
            axis = temp_axis / (temp_axis.pow(2).sum().pow(1/2) + 1e-8 )
            temp = (torch.norm(delta_t_k,dim=0) + 1e-8) * (torch.norm(delta_p_k,dim=0) + 1e-8)
            cos_alpha = torch.dot(delta_t_k, delta_p_k) / temp

            alpha = torch.acos(cos_alpha)

            twist = delta_t_k
            D_sw = axangle2mat(axis=axis, angle=alpha, is_normalized=False)
            D_tw = axangle2mat(axis=twist, angle=angels0[:, k], is_normalized=False)

            R_pa_k[k] = torch.matmul(D_sw, D_tw)
            R[k] = torch.matmul(R[pa], R_pa_k[k]).to(torch.float32)

        pose_R = torch.zeros((1, 16, 3, 3)).to(T_.device)
        pose_R[0, 0] = R[0]
        for key in cfg.ID2ROT.keys():
            value = cfg.ID2ROT[key]
            pose_R[0, value] = R_pa_k[key]
        
        pose_R = rotation_to_axis_angle(pose_R)

        if idx == 0:
            gt_rotatiation = pose_R.reshape(1,-1)
        else:
            gt_rotatiation = torch.cat([gt_rotatiation, pose_R.reshape(1,-1)],dim=0)

    return gt_rotatiation

