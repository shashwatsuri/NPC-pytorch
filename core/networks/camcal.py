import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from core.utils.skeleton_utils import *
from typing import Optional, Any



def whiten(camvec9d,P1):
    return camvec9d
    # return torch.matmul(P1,camvec9d.T).T

def dewhiten(camvec9d,invP1):
    return camvec9d
    # return torch.matmul(invP1,camvec9d.T).T


def add_normal_noise(self, tensor, std=0.005):
    """
    Add random noise from a normal distribution to a PyTorch tensor.

    Parameters:
    - tensor (torch.Tensor): Input tensor to which noise will be added.
    - mean (float): Mean of the normal distribution (default is 0).
    - std (float): Standard deviation of the normal distribution (default is 0.005).

    Returns:
    - torch.Tensor: Tensor with added noise.
    """
    noise = torch.randn_like(tensor) * std
    noisy_tensor = tensor + noise
    assert tensor.size() == noisy_tensor.size()
    return noisy_tensor

def zca_matrix(self,X):
    # Calculate covariance matrix
    X=X.unsqueeze(0)
    covariance_matrix = torch.matmul(X.t(), X)
    # Perform eigen decomposition
    U, S, _ = torch.svd(covariance_matrix)
    # Apply ZCA whitening transformation
    epsilon = 1e-5  # Small constant to avoid division by zero
    X_zca = torch.matmul(torch.matmul(U, torch.diag(1.0 / torch.sqrt(S + epsilon))), U.t())

    return X_zca


#TODO: OPT_T ALWAYS ASSUMED TO BE TRUE
class CamCal(nn.Module):

    def __init__(
        self,
        P,
        invP,
        n_cams: int = 3,
        load_path: Optional[str] = None,
        stop_opt: bool = False,
        opt_T: bool = True,
        error: float = 0.0,
    ):
        super().__init__()
        self.n_cams = n_cams
        self.load_path = load_path
        self.stop_opt = stop_opt
        self.opt_T = opt_T
        R = torch.eye(3)[None] #([1,3,3])
        Rvec = rot_to_rot6d(R).expand(n_cams, -1) #([3,6])
        if self.load_path is not None:
            device = Rvec.device
            Rvec = torch.load(load_path, map_location=device)
        T = torch.zeros(3)[None]
        T = T.expand(n_cams, -1) #([3,3])
        op_camvec = torch.zeros(3,9)
        for i in range(len(T)):
            camvec9d = torch.concat((Rvec[i],T[i]))[None]
            op_camvec9d = whiten(camvec9d,P[i])[0]
            op_camvec[i] = op_camvec9d
        print("successfully initialized cam_cal")
        # self.register_parameter('Rvec', nn.Parameter(Rvec.clone(), requires_grad=not self.stop_opt))
        # self.register_parameter('T', nn.Parameter(T.clone(), requires_grad=not self.stop_opt))
        self.register_parameter('op_camvec', nn.Parameter(op_camvec.clone(), requires_grad=not self.stop_opt))
        self.register_buffer('P',P)
        self.register_buffer('invP',invP)

    
    
    
    def forward(
        self,
        batch: dict,
        z_vals: torch.Tensor,
        **kwargs,
    ):
        # print("im forwarding as well")
        if 'real_cam_idx' not in batch:
            cam_idxs = torch.zeros(z_vals.shape[0]).long() + self.identity_cam
        else:
            cam_idxs = batch['real_cam_idx']
        
        # Rvec = self.Rvec
        # T = self.T

        op_camvec = self.op_camvec
        P = self.P
        invP = self.invP

        Rvec = torch.zeros((3,6))
        T = torch.zeros((3,3))
        
        for i in range(len(op_camvec)):
            camvec = dewhiten(op_camvec[i],invP[i])
            Rvec[i] = camvec[:6]
            T[i] = camvec[6:]

        R = rot6d_to_rotmat(Rvec)
        # masks = (cam_idxs == self.identity_cam).float()
        # masks = masks.reshape(-1, 1, 1)
        # identity = torch.eye(3)[None]
        # breakpoint()
        R = R[cam_idxs]

        rays_o = batch['rays_o']
        rays_d = batch['rays_d']

        # adjust the ray
        rays_d_cal = (rays_d[:, None] @ R)
        # masks = masks.reshape(-1, 1) 
        # identity = torch.zeros(3)[None]
        T = T[cam_idxs]
        rays_o_cal = rays_o[:, None] + T[:, None]
        pts_cal = rays_d_cal * z_vals[..., None] + rays_o_cal

        batch.update(
            pts=pts_cal,
            rays_d=rays_d_cal[:, 0],
        )

        return batch

class ColorCal(nn.Module):

    def __init__(
        self,
        n_cams: int = 4,
        identity_cam: int = 0,
        load_path: Optional[str] = None,
        stop_opt: bool = False,
    ):
        super().__init__()
        self.n_cams = n_cams
        self.identity_cam = identity_cam
        self.load_path = load_path
        self.stop_opt = stop_opt

        cal = torch.tensor([[1., 1., 1., 0., 0., 0.]]).expand(n_cams, -1)

        if self.load_path is not None:
            device = cal.device
            cal = torch.load(load_path, map_location=device)

        self.register_parameter('cal', nn.Parameter(cal.clone(), requires_grad=not self.stop_opt))
    
    def forward(
        self,
        batch: dict,
        rgb_map: torch.Tensor,
        **kwargs,
    ):
        if batch is None or 'real_cam_idx' not in batch:
            cam_idxs = torch.zeros(rgb_map.shape[0]).long() + self.identity_cam
        else:
            cam_idxs = batch['real_cam_idx']

        cal = self.cal
        if self.load_path is not None:
            cal = cal.detach()
 
        masks = (cam_idxs == self.identity_cam).float()
        masks = masks.reshape(-1, 1)
        identity = torch.tensor([[1., 1., 1., 0., 0., 0.]])

        cal = cal[cam_idxs] * (1 - masks) + identity * masks
        rgb_cal = rgb_map * cal[:, :3] + cal[:, 3:] 

        return rgb_cal
