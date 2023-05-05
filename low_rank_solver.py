# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import os
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from scipy import rand
import torchvision.transforms as transforms
from torch import _linalg_utils as _utils
import torch
import numpy as np
import torch
from PIL import Image, ImageOps
import kornia
from math import pi
import uuid
from utils import *


def init_svd(input_size):
    A = torch.randn(input_size, input_size)
    U, S, Ut = torch.linalg.svd(A)
    return U, S, Ut

def get_approximate_basis(
    A: torch.Tensor, q: int, niter: int = 2, M: torch.Tensor = None
) -> torch.Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = _utils.get_floating_dtype(A)
    matmul = _utils.matmul

    R = torch.randn(n, q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable
    A_H = _utils.transjugate(A)
    if M is None:
        Q = torch.linalg.qr(matmul(A, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q)).Q
    else:
        M_H = _utils.transjugate(M)
        Q = torch.linalg.qr(matmul(A, R) - matmul(M, R)).Q
        for i in range(niter):
            Q = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q)).Q
            Q = torch.linalg.qr(matmul(A, Q) - matmul(M, Q)).Q

    return Q

def _svd_lowrank(
    A: torch.Tensor,
    q: int = 6,
    niter: int = 2,
    M: torch.Tensor = None,
) -> torch.Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is None:
        M_t = None
    else:
        M_t = _utils.transpose(M)
    A_t = _utils.transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in
        # order to keep B shape minimal (the m < n case) or the V
        # shape small (the n > q case)
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = _utils.transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, Vh = torch.linalg.svd(B_t, full_matrices=False)
        V = Vh.mH
        U = Q.matmul(U)

    return U, S, V

def _svd_lowrank_by_channel(delta, rank):
    matmul = _utils.matmul
    
    delta = torch.squeeze(delta, 0)
    U_list = []
    S_list = []
    V_list = []

    for i in range(3):
        U, S, V = _svd_lowrank(delta[i], q = rank) 
        # U, S, V = torch.linalg.svd(delta[i], full_matrices=False)
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)

    return torch.stack(U_list, dim=0), torch.stack(S_list, dim=0), torch.stack(V_list, dim=0)

def _svd_lowrank_by_batch_channel(delta, rank):

    matmul = _utils.matmul

    batch_U_list = []
    batch_S_list = []
    batch_V_list = []

    for j in range(delta.shape[0]):    
        U_list = []
        S_list = []
        V_list = [] 
        for i in range(3):
            U, S, V = _svd_lowrank(delta[j][i], q = rank) 
            # U, S, V = torch.linalg.svd(delta[j][i], full_matrices=False)
            U_list.append(U)
            S_list.append(S)
            V_list.append(V)
        batch_U_list.append(torch.stack(U_list, dim=0))
        batch_S_list.append(torch.stack(S_list, dim=0))
        batch_V_list.append(torch.stack(V_list, dim=0))
    return torch.stack(batch_U_list, dim=0), torch.stack(batch_S_list, dim=0), torch.stack(batch_V_list, dim=0)

def svd_lowrank_transform(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, rank: int) -> torch.Tensor:
    
    matmul = _utils.matmul
    
    # if not isinstance(rank, torch.Tensor):
    #     rank = torch.as_tensor(rank, device=input.device, dtype=input.dtype)

    I = torch.eye(U[0].shape[0]).to(device)
    # print(I)
    # print(matmul(U[0], _utils.transpose(U[0])))
    # print(torch.eq(matmul(U[0], _utils.transpose(U[0])), I))
    # if torch.eq(matmul(U[0], _utils.transpose(V[0])), I):
    #     V[0] = U[0]
    # if torch.eq(matmul(U[1], _utils.transpose(V[1])), I):
    #     V[1] = U[1]
    # if torch.eq(matmul(U[2], _utils.transpose(V[2])), I):
    #     V[2] = U[2]
    r_delta = matmul(matmul(U[0], torch.diag(S[0])), _utils.transpose(V[0]))
    # rU, rS, rV = torch.linalg.svd(r_delta, full_matrices=False)
    # rU, rS, rV = _svd_lowrank(r_delta, q = rank) 
    # lowrank_r_delta = matmul(matmul(rU, torch.diag(rS)), _utils.transpose(rV))
    # print('diff: ', lowrank_r_delta - r_delta)
    g_delta = matmul(matmul(U[1], torch.diag(S[1])), _utils.transpose(V[1]))
    # gU, gS, gV = _svd_lowrank(g_delta, q = rank) 
    # lowrank_g_delta = matmul(matmul(gU, torch.diag(gS)), _utils.transpose(gV))
    b_delta = matmul(matmul(U[2], torch.diag(S[2])), _utils.transpose(V[2]))   
    # bU, bS, bV = _svd_lowrank(b_delta, q = rank) 
    # lowrank_b_delta = matmul(matmul(bU, torch.diag(bS)), _utils.transpose(bV))

    return torch.stack([r_delta, g_delta, b_delta], dim = 0)

def svd_lowrank_transform_bybatch(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, rank: int) -> torch.Tensor:
    
    matmul = _utils.matmul
    
    # if not isinstance(rank, torch.Tensor):
    #     rank = torch.as_tensor(rank, device=input.device, dtype=input.dtype)
    # print(I)
    # print(matmul(U[0], _utils.transpose(U[0])))
    # print(torch.eq(matmul(U[0], _utils.transpose(U[0])), I))
    # if torch.eq(matmul(U[0], _utils.transpose(V[0])), I):
    #     V[0] = U[0]
    # if torch.eq(matmul(U[1], _utils.transpose(V[1])), I):
    #     V[1] = U[1]
    # if torch.eq(matmul(U[2], _utils.transpose(V[2])), I):
    #     V[2] = U[2]
    b_size = U.shape[0]
    batch_lowrank_delta = []
    for i in range(b_size):

        r_delta = matmul(matmul(U[i][0], torch.diag(S[i][0])), _utils.transpose(V[i][0]))
        # rU, rS, rV = _svd_lowrank(r_delta, q = rank) 
        # lowrank_r_delta = matmul(matmul(rU, torch.diag(rS)), _utils.transpose(rV))

        g_delta = matmul(matmul(U[i][1], torch.diag(S[i][1])), _utils.transpose(V[i][1]))
        # gU, gS, gV = _svd_lowrank(g_delta, q = rank) 
        # lowrank_g_delta = matmul(matmul(gU, torch.diag(gS)), _utils.transpose(gV))

        b_delta = matmul(matmul(U[i][2], torch.diag(S[i][2])), _utils.transpose(V[i][2]))   
        # bU, bS, bV = _svd_lowrank(b_delta, q = rank) 
        # lowrank_b_delta = matmul(matmul(bU, torch.diag(bS)), _utils.transpose(bV))
        
        batch_lowrank_delta.append(torch.stack([r_delta, g_delta, b_delta], dim = 0))
    return torch.stack(batch_lowrank_delta, dim = 0)

def _blend_one(input1: torch.Tensor, input2: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    r"""Blend two images into one.

    Args:
        input1: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        input2: image tensor with shapes like :math:`(H, W)` or :math:`(D, H, W)`.
        factor: factor 0-dim tensor.

    Returns:
        : image tensor with the batch in the zero position.
    """
    if not isinstance(input1, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input1}.")
    if not isinstance(input2, torch.Tensor):
        raise AssertionError(f"`input1` must be a tensor. Got {input2}.")

    if isinstance(factor, torch.Tensor) and len(factor.size()) != 0:
        raise AssertionError(f"Factor shall be a float or single element tensor. Got {factor}.")
    if factor == 0.0:
        return input1
    if factor == 1.0:
        return input2
    diff = (input2 - input1) * factor
    res = input1 + diff
    if factor > 0.0 and factor < 1.0:
        return res
    return torch.clamp(res, 0, 1)



def get_transAug_param(aug_name, corr_type):
    
    noise_list = ['gaussian_noise', 'impulse_noise', 'shot_noise', 'brightness', 'snow', 'frost']
    if aug_name == 'contrast':
        contrast_epsilon = torch.tensor((0, 2))
        return contrast_epsilon
    elif aug_name == 'brightness':
        brightness_epsilon = torch.tensor((0, 1))
        return brightness_epsilon
    elif aug_name == 'hue':
        hue_epsilon = torch.tensor((-pi/3, pi/3))
        return hue_epsilon
    elif aug_name == 'saturation':
        sat_epsilon = torch.tensor((0, 1))
        return sat_epsilon
    elif aug_name == 'sharpness':
        if corr_type in noise_list:
            sharp_epsilon = torch.tensor((0.5, 1))
        else: sharp_epsilon = torch.tensor((0.5, 3)) 
        return sharp_epsilon
    elif aug_name == 'solarize':
        solar_epsilon = torch.tensor((0.5, 0.7))
        return solar_epsilon

def get_imgnet_transaug_param(aug_name):
  
    if aug_name == 'sharpness':
        sharp_epsilon = torch.tensor((0.5, 1))
        return sharp_epsilon