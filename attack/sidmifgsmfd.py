import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def resize_and_pad(x, scale):
    B, C, H, W = x.shape
    new_h = int(H * scale)
    new_w = int(W * scale)

    x_small = F.interpolate(
        x, size=(new_h, new_w),
        mode='bilinear', align_corners=False
    )

    pad_h = H - new_h
    pad_w = W - new_w

    top = random.randint(0, pad_h)
    left = random.randint(0, pad_w)
    bottom = pad_h - top
    right = pad_w - left

    return F.pad(x_small, (left, right, top, bottom), value=0)

def random_horizontal_flip(x, p=0.5):
    if random.random() < p:
        return torch.flip(x, dims=[3])
    return x

def linear_fusion(block, global_resized, omega):
    return omega * global_resized + (1 - omega) * block

def dct_2d(x):
    return torch.fft.fft2(x, norm='ortho')

def idct_2d(x):
    return torch.fft.ifft2(x, norm='ortho').real

def frequency_fusion(block, global_resized, low_ratio=0.6):
    B, C, H, W = block.shape
    dct_block = dct_2d(block)
    dct_global = dct_2d(global_resized)

    mask = torch.zeros_like(dct_block)
    h_cut = int(H * low_ratio)
    w_cut = int(W * low_ratio)
    mask[:, :, :h_cut, :w_cut] = 1.0

    low = dct_block * mask
    high = dct_global * (1 - mask)
    return idct_2d(low + high)

def local_image_fusion(x, k, p, omega):
    B, C, H, W = x.shape
    h_blk, w_blk = H // k, W // k
    out = x.clone()

    for i in range(k):
        for j in range(k):
            if random.random() > p:
                continue

            h1, h2 = i*h_blk, (i+1)*h_blk
            w1, w2 = j*w_blk, (j+1)*w_blk

            block = x[:, :, h1:h2, w1:w2]
            global_resized = F.interpolate(
                x, size=(h_blk, w_blk),
                mode='bilinear', align_corners=False
            )

            if random.random() < 0.5:
                fused = linear_fusion(block, global_resized, omega)
            else:
                fused = frequency_fusion(block, global_resized)

            out[:, :, h1:h2, w1:w2] = fused

    return out

def sid_transform(x, beta, n, N, k, p, omega):
    scale = max(1.0 - beta * n / N, 0.3)
    x = resize_and_pad(x, scale)
    x = random_horizontal_flip(x)
    x = local_image_fusion(x, k, p, omega)
    return x

class LinfFDSIDMIFGSMAttack:
    def __init__(
        self,
        args,
        model,
        device,
        eps=16/255,
        alpha=2/255,
        steps=10,
        decay=1.0,
        N=20,
        beta=0.1,
        k=2,
        p=0.5,
        omega=0.5
    ):
        self.model = model
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        # SID parameters
        self.N = N
        self.beta = beta
        self.k = k
        self.p = p
        self.omega = omega
        self.weight = args.weight
        self.regularization = args.regularization

    def perturb(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.regularization == 'CE':
            regularization = nn.CrossEntropyLoss()
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)

            for _ in range(self.steps):
                grad_sum = torch.zeros_like(adv_images)

                for n in range(1, self.N + 1):
                    x_t = sid_transform(
                        adv_images, self.beta, n, self.N,
                        self.k, self.p, self.omega
                    )
                    x_t.requires_grad_(True)

                    adv_outputs = self.model(x_t)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    cost = loss(adv_outputs, labels) + self.weight * regularization(adv_outputs, benign_outputs)
                    # loss = F.cross_entropy(adv_outputs, y)
                    grad = torch.autograd.grad(cost, x_t)[0]

                    grad_sum += grad

                grad = grad_sum / self.N
                grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)

                momentum = self.decay * momentum + grad
                adv_images = adv_images + self.alpha * torch.sign(momentum)

                adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
                adv_images = torch.clamp(adv_images, 0, 1).detach()

        elif self.regularization == 'MSE':
            regularization = nn.MSELoss()
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)

            for _ in range(self.steps):
                grad_sum = torch.zeros_like(adv_images)

                for n in range(1, self.N + 1):
                    x_t = sid_transform(
                        adv_images, self.beta, n, self.N,
                        self.k, self.p, self.omega
                    )
                    x_t.requires_grad_(True)

                    adv_outputs = self.model(x_t)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    cost = loss(adv_outputs, labels) + self.weight * regularization(adv_outputs, benign_outputs)
                    # loss = F.cross_entropy(adv_outputs, y)
                    grad = torch.autograd.grad(cost, x_t)[0]

                    grad_sum += grad

                grad = grad_sum / self.N
                grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)

                momentum = self.decay * momentum + grad
                adv_images = adv_images + self.alpha * torch.sign(momentum)

                adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
                adv_images = torch.clamp(adv_images, 0, 1).detach()

        elif self.regularization == 'KL':
            regularization = nn.KLDivLoss(reduction="batchmean")
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)

            for _ in range(self.steps):
                grad_sum = torch.zeros_like(adv_images)

                for n in range(1, self.N + 1):
                    x_t = sid_transform(
                        adv_images, self.beta, n, self.N,
                        self.k, self.p, self.omega
                    )
                    x_t.requires_grad_(True)

                    adv_outputs = self.model(x_t)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    log_adv_outputs = torch.log(adv_outputs)
                    cost = loss(adv_outputs, labels) + self.weight * regularization(log_adv_outputs, benign_outputs)
                    # loss = F.cross_entropy(adv_outputs, y)
                    grad = torch.autograd.grad(cost, x_t)[0]

                    grad_sum += grad

                grad = grad_sum / self.N
                grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)

                momentum = self.decay * momentum + grad
                adv_images = adv_images + self.alpha * torch.sign(momentum)

                adv_images = torch.min(torch.max(adv_images, images - self.eps), images + self.eps)
                adv_images = torch.clamp(adv_images, 0, 1).detach()

        return adv_images
