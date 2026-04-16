import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------
# 工具函数：构建旋转矩阵
# ------------------------------------------------------------
def _make_rotation_theta(angles, device):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    theta = torch.zeros((angles.size(0), 2, 3), device=device, dtype=angles.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    return theta


# ------------------------------------------------------------
# 工具函数：旋转 patch
# ------------------------------------------------------------
def rotate_patches(patches, angles):
    """
    patches: (M, C, ph, pw)
    angles:  (M,)
    """
    M, C, ph, pw = patches.shape
    device = patches.device
    theta = _make_rotation_theta(angles, device)
    grid = F.affine_grid(theta, size=(M, C, ph, pw), align_corners=False)
    rotated = F.grid_sample(
        patches, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )
    return rotated


# ------------------------------------------------------------
# Block Shuffle + Rotation
# ------------------------------------------------------------
def block_shuffle_and_rotate(images, n_blocks=2, max_angle=24):
    """
    images: (B,C,H,W)
    返回变换后的图像（可反传）
    """
    B, C, H, W = images.shape
    device = images.device
    nb = n_blocks
    ph, pw = H // nb, W // nb

    patches = images.view(B, C, nb, ph, nb, pw).permute(0,2,4,1,3,5).contiguous()
    patches = patches.view(B, nb*nb, C, ph, pw)
    M = nb * nb
    patches_flat = patches.view(B*M, C, ph, pw)

    out_imgs = torch.zeros_like(images)

    for i in range(B):
        perm = torch.randperm(M, device=device)

        angles = (torch.rand(M, device=device) * 2 - 1) * max_angle
        angles = angles * math.pi / 180.0

        this = patches_flat[i*M:(i+1)*M]
        rotated = rotate_patches(this, angles)
        permuted = rotated[perm]

        permuted = permuted.view(nb, nb, C, ph, pw).permute(2,0,3,1,4).contiguous()
        out_imgs[i] = permuted.view(C, H, W)

    return out_imgs


# ------------------------------------------------------------
# BSR + MI-FGSM 实现
# ------------------------------------------------------------
class LinfBSRAttack(object):
    def __init__(
        self, model, device, eps=16/255, alpha=2/255, steps=10, decay=1.0,
        n_blocks=2, max_angle=24, N=20
    ):
        self.model = model
        self.device = device

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        self.n_blocks = n_blocks
        self.max_angle = max_angle
        self.N = N


    @torch.no_grad()
    def forward(self, x):
        return self.model(x)

    def perturb(self, images, labels):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(images).to(self.device)

        for t in range(self.steps):
            adv_images.requires_grad = True

            # ---- 关键：累积 N 个变换后的梯度 ---- #
            grad_accum = torch.zeros_like(images)

            for _ in range(self.N):
                images_trans = block_shuffle_and_rotate(
                    adv_images, n_blocks=self.n_blocks, max_angle=self.max_angle
                )

                adv_outputs = self.model(images_trans)
                adv_outputs = F.softmax(adv_outputs, dim=1)
                cost = loss(adv_outputs, labels)
                # loss = self.criterion(adv_outputs, labels)

                grad = torch.autograd.grad(cost, adv_images, retain_graph=True)[0]
                grad_accum += grad.detach()

            # 平均梯度
            g = grad_accum / self.N
            g = g / (g.abs().mean(dim=(1,2,3), keepdim=True) + 1e-8)

            # 动量
            momentum = self.decay * momentum + g

            # 更新扰动
            adv_images = adv_images + self.alpha * torch.sign(momentum)
            adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
            adv_images.requires_grad = True

        return adv_images.detach()
