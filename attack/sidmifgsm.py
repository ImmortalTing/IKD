import torch
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

class LinfSIDMIFGSMAttack:
    def __init__(
        self,
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

    def perturb(self, x, y):
        x_adv = x.clone().detach()
        g = torch.zeros_like(x_adv)

        for _ in range(self.steps):
            grad_sum = torch.zeros_like(x_adv)

            for n in range(1, self.N + 1):
                x_t = sid_transform(
                    x_adv, self.beta, n, self.N,
                    self.k, self.p, self.omega
                )
                x_t.requires_grad_(True)

                logits = self.model(x_t)
                loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, x_t)[0]

                grad_sum += grad

            grad = grad_sum / self.N
            grad = grad / (torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) + 1e-8)

            g = self.decay * g + grad
            x_adv = x_adv + self.alpha * torch.sign(g)

            x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1).detach()

        return x_adv
