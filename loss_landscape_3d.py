import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 支持

from attack_main import create_attackers, MODEL_CONFIG
import timm
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_loss_landscape(model, base_img, target_label, directions, grid_size=21, scale=0.1):
    xs = np.linspace(-scale, scale, grid_size)
    ys = np.linspace(-scale, scale, grid_size)
    loss_values = np.zeros((grid_size, grid_size))
    loss_fn = torch.nn.CrossEntropyLoss()

    for i, alpha in enumerate(xs):
        for j, beta in enumerate(ys):
            perturbed = base_img + alpha * directions[0] + beta * directions[1]
            perturbed = torch.clamp(perturbed, 0, 1)

            with torch.no_grad():
                out = model(perturbed)
                loss = loss_fn(out, target_label)
                loss_values[j, i] = loss.item()

    return xs, ys, loss_values

# -------------------- 数据准备 --------------------

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

img_path = './data/dev_data/val_rs/ILSVRC2012_val_00000036.JPEG'
label_val = 888  # 假设真实类是 888
label = torch.tensor([label_val - 1])  # ImageNet 标签从 0 开始

img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# -------------------- 模型 & 攻击器 --------------------

model = timm.create_model(MODEL_CONFIG['resnet50'], pretrained=True).eval().to(device)

args = type('', (), {})()
args.attack = 'mifgsm'
args.weight = 0.01
args.regularization = 'KL'

attacker, fdattacker = create_attackers(args, model, device)

adv_img = attacker.perturb(img_tensor, label.to(device))
fdadv_img = fdattacker.perturb(img_tensor, label.to(device))

# -------------------- 采样方向 --------------------

dir1 = torch.randn_like(img_tensor).to(device)
dir2 = torch.randn_like(img_tensor).to(device)
dir1 = dir1 / (dir1.norm() + 1e-10)
dir2 = dir2 / (dir2.norm() + 1e-10)

# -------------------- 采样 loss surface --------------------

grid = 42
scale = 100.0

xs, ys, loss_before = sample_loss_landscape(model, adv_img, label.to(device), (dir1, dir2), grid_size=grid, scale=scale)
_, _, loss_after = sample_loss_landscape(model, fdadv_img, label.to(device), (dir1, dir2), grid_size=grid, scale=scale)

X, Y = np.meshgrid(xs, ys)

# -------------------- 3D 可视化 --------------------

fig = plt.figure(figsize=(14, 6))

# 正则化前
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, loss_before, cmap='viridis')
ax1.set_title('3D Loss Surface (No Regularization)')
ax1.set_xlabel('Direction 1 scale')
ax1.set_ylabel('Direction 2 scale')
ax1.set_zlabel('Loss')

# 正则化后
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, loss_after, cmap='viridis')
ax2.set_title('3D Loss Surface (With Regularization)')
ax2.set_xlabel('Direction 1 scale')
ax2.set_ylabel('Direction 2 scale')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.savefig('loss_landscape_3d_comparison.png')
plt.close()
