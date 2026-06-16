import argparse
import os
import random

import timm
from torch.utils import data
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from attack.mifgsm import LinfMIFGSMAttack
from attack.mifgsmfd import LinfFDMIFGSMAttack
from attack.difgsm import LinfDIFGSMAttack
from attack.difgsmfd import LinfFDDIFGSMAttack
from attack.tifgsm import LinfTIFGSMAttack
from attack.tifgsmfd import LinfFDTIFGSMAttack
from attack.nifgsm import LinfNIFGSMAttack
from attack.nifgsmfd import LinfFDNIFGSMAttack
from attack.sinifgsm import LinfSINIFGSMAttack
from attack.sinifgsmfd import LinfFDSINIFGSMAttack
from attack.vmifgsm import LinfVMIFGSMAttack
from attack.vmifgsmfd import LinfFDVMIFGSMAttack
from attack.vnifgsm import LinfVNIFGSMAttack
from attack.vnifgsmfd import LinfFDVNIFGSMAttack
from attack.bsrmifgsm import LinfBSRAttack
from attack.bsrmifgsmfd import LinfFDBSRAttack
from attack.sidmifgsm import LinfSIDMIFGSMAttack
from attack.sidmifgsmfd import LinfFDSIDMIFGSMAttack
from attack.ggsmifgsm import LinfGGSMIFGSMAttack
from attack.ggsmifgsmfd import LinfFDGGSMIFGSMAttack


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dev_data")
IMAGE_DIR = os.path.join(DATA_DIR, "val_rs")
CSV_PATH = os.path.join(DATA_DIR, "val_rs.csv")

# ---------------------------- 模型配置 ----------------------------
MODEL_CONFIG = {
    'resnet50': 'resnet50',
    'densenet121': 'densenet121',
    'resnext50': 'resnext50_32x4d',
    'vgg19bn': 'vgg19_bn',
    'incres_v2': 'inception_resnet_v2',
    'inc_v3': 'inception_v3',
    'inc_v4': 'inception_v4',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    'adv_inception_v3': 'adv_inception_v3',
    'ens_adv_inception_resnet_v2': 'ens_adv_inception_resnet_v2',
    'visformer_small': 'visformer_small',
    'vit_b': 'vit_base_patch16_224',
    'swin_b': 'swin_s3_base_224',
    'pit_b': 'pit_b_224',
    'mobilenet': 'mobilenetv2_050',
}



# ---------------------------- 攻击类配置 ----------------------------
ATTACK_CLASSES = {
    'mifgsm': {'base': LinfMIFGSMAttack, 'ikd': LinfFDMIFGSMAttack},
    'difgsm': {'base': LinfDIFGSMAttack, 'ikd': LinfFDDIFGSMAttack},
    'tifgsm': {'base': LinfTIFGSMAttack, 'ikd': LinfFDTIFGSMAttack},
    'nifgsm': {'base': LinfNIFGSMAttack, 'ikd': LinfFDNIFGSMAttack},
    'sinifgsm': {'base': LinfSINIFGSMAttack, 'ikd': LinfFDSINIFGSMAttack},
    'vmifgsm': {'base': LinfVMIFGSMAttack, 'ikd': LinfFDVMIFGSMAttack},
    'vnifgsm': {'base': LinfVNIFGSMAttack, 'ikd': LinfFDVNIFGSMAttack},
    'bsrmifgsm': {'base': LinfBSRAttack, 'ikd': LinfFDBSRAttack},
    'sidmifgsm': {'base': LinfSIDMIFGSMAttack, 'ikd': LinfFDSIDMIFGSMAttack},
    'ggsmifgsm': {'base': LinfGGSMIFGSMAttack, 'ikd': LinfFDGGSMIFGSMAttack},
}



# ---------------------------- 攻击参数 ----------------------------
COMMON_ATTACK_PARAMS = {
    'eps': 16/255,
    'alpha': 2/255,
    'steps': 10,
    'decay': 1.0
}

SPECIFIC_ATTACK_PARAMS = {
    'difgsm': {'resize_rate': 0.9, 'diversity_prob': 0.5, 'random_start': False},
    'tifgsm': {
        'kernel_name': 'gaussian', 'len_kernel': 15, 'nsig': 3,
        'resize_rate': 0.9, 'diversity_prob': 0.5, 'random_start': False
    },
    'sinifgsm': {'m': 5},
    'vmifgsm': {'N': 5, 'beta': 3/2},
    'bsrmifgsm': {'n_blocks': 2, 'max_angle': 24, 'N': 20},
    'sidmifgsm': {'N': 20, 'beta': 0.1, 'k': 2, 'p': 0.5, 'omega': 0.5},
    'ggsmifgsm': {'N': 20, 'zeta': 2.0},
}



# ---------------------------- 设置随机种子 ----------------------------
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# # ---------------------------- ImageNet 数据集 ----------------------------
# class ImagenetDataset(data.Dataset):
#     def __init__(self, transform):
#         img_id = []
#         label = []
#         self.file_path = './data/dev_data/val_rs/'
#         f = open("./data/dev_data/val_rs.csv", "r")
#         label_dict = f.read().splitlines()

#         for i in range(1, len(label_dict)):
#             row = label_dict[i].split(",")
#             img_id.append(row[0])
#             label.append(row[1])
#         f.close()

#         self.img_id = img_id
#         self.label = label
#         self.transform = transform

#     def __getitem__(self, index):
#         label = int(self.label[index - 1]) - 1
#         img_id = self.img_id[index - 1]

#         img_path = self.file_path + img_id
#         img = np.array(Image.open(img_path))

#         if self.transform:
#             img = self.transform(img)

#         return img, label, img_id

#     def __len__(self):
#         return len(self.label)

class ImagenetDataset(data.Dataset):
    def __init__(self, transform):
        img_id = []
        label = []

        with open(CSV_PATH, "r") as handle:
            label_rows = handle.read().splitlines()

        for i in range(1, len(label_rows)):
            row = label_rows[i].split(",")
            img_id.append(row[0])
            label.append(row[1])

        self.img_id = img_id
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        label = int(self.label[index]) - 1
        img_id = self.img_id[index]

        img_path = os.path.join(IMAGE_DIR, img_id)
        with Image.open(img_path) as pil_image:
            img = pil_image.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, img_id

    def __len__(self):
        return len(self.label)



# ---------------------------- 创建攻击器 ----------------------------
def create_attackers(args, model, device):
    attack_params = {**COMMON_ATTACK_PARAMS}
    if args.attack in SPECIFIC_ATTACK_PARAMS:
        attack_params.update(SPECIFIC_ATTACK_PARAMS[args.attack])

    attack_cls = ATTACK_CLASSES[args.attack]
    return (
        attack_cls['base'](model=model, device=device, **attack_params),
        attack_cls['ikd'](args, model=model, device=device, **attack_params)
    )


def append_result_row(handle, model_name, data):
    handle.write(
        f"{model_name}_benign_acc: {data['benign_acc']:.4f}, "
        f"{model_name}_benign_asr: {100 - data['benign_acc']:.4f}, "
        f"{model_name}_adv_acc: {data['adv_acc']:.4f}, "
        f"{model_name}_adv_asr: {100 - data['adv_acc']:.4f}, "
        f"{model_name}_fdadv_acc: {data['fdadv_acc']:.4f}, "
        f"{model_name}_fdadv_asr: {100 - data['fdadv_acc']:.4f}\n"
    )
    handle.flush()


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_eval_cache(args, dataloader, device):
    print(f"\nLoading SOURCE model: {args.model} ...")
    source_model = freeze_model(
        timm.create_model(MODEL_CONFIG[args.model], pretrained=True).to(device)
    )
    adversary, ikd_adversary = create_attackers(args, source_model, device)

    use_cuda = device == 'cuda'
    cached_batches = []

    for inputs, targets, _ in tqdm(dataloader, desc="Generating attack cache"):
        if use_cuda and not inputs.is_pinned():
            cached_inputs = inputs.pin_memory()
        else:
            cached_inputs = inputs

        if use_cuda and not targets.is_pinned():
            cached_targets = targets.pin_memory()
        else:
            cached_targets = targets

        inputs = cached_inputs.to(device, non_blocking=use_cuda)
        targets = cached_targets.to(device, non_blocking=use_cuda)

        with torch.enable_grad():
            adv = adversary.perturb(inputs, targets)
            ikd_adv = ikd_adversary.perturb(inputs, targets)

        cached_adv = adv.detach().cpu()
        cached_ikd_adv = ikd_adv.detach().cpu()
        if use_cuda:
            cached_adv = cached_adv.pin_memory()
            cached_ikd_adv = cached_ikd_adv.pin_memory()

        cached_batches.append(
            (cached_inputs, cached_adv, cached_ikd_adv, cached_targets)
        )

        del inputs, targets, adv, ikd_adv

    del source_model, adversary, ikd_adversary
    if use_cuda:
        torch.cuda.empty_cache()

    return cached_batches


def evaluate_cached_batches(target_model, cached_batches, device):
    use_cuda = device == 'cuda'
    benign_correct = 0
    adv_correct = 0
    ikd_adv_correct = 0

    with torch.inference_mode():
        for inputs_cpu, adv_cpu, ikd_adv_cpu, targets_cpu in tqdm(cached_batches):
            inputs = inputs_cpu.to(device, non_blocking=use_cuda)
            adv = adv_cpu.to(device, non_blocking=use_cuda)
            ikd_adv = ikd_adv_cpu.to(device, non_blocking=use_cuda)
            targets = targets_cpu.to(device, non_blocking=use_cuda)

            outputs = target_model(inputs)
            _, pred = outputs.max(1)
            benign_correct += pred.eq(targets).sum().item()

            outputs = target_model(adv)
            _, pred = outputs.max(1)
            adv_correct += pred.eq(targets).sum().item()

            outputs = target_model(ikd_adv)
            _, pred = outputs.max(1)
            ikd_adv_correct += pred.eq(targets).sum().item()

            del inputs, adv, ikd_adv, targets, outputs, pred

    return benign_correct, adv_correct, ikd_adv_correct



# ---------------------------- 主测试流程（已修复显存） ----------------------------
def test(args, out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total = len(imagenet_loader.dataset)
    model_names = list(MODEL_CONFIG.keys())
    results = {}
    cached_batches = build_eval_cache(args, imagenet_loader, device)

    # -----------------------------------------
    # ✔️ 针对每个目标模型进行测试（逐个加载/释放）
    # -----------------------------------------
    with open(out_path, 'a') as result_file:
        for target_name in model_names:
            print(f"\n========== Testing target model: {target_name} ==========")

            target_model = freeze_model(
                timm.create_model(MODEL_CONFIG[target_name], pretrained=True).to(device)
            )
            benign_correct, adv_correct, ikd_adv_correct = evaluate_cached_batches(
                target_model, cached_batches, device
            )

            # 保存结果
            results[target_name] = {
                'benign_acc': 100. * benign_correct / total,
                'adv_acc': 100. * adv_correct / total,
                'fdadv_acc': 100. * ikd_adv_correct / total
            }
            append_result_row(result_file, target_name, results[target_name])

            # ✔️ 释放显存
            del target_model
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f"Released GPU memory for model: {target_name}")

    # 打印结果
    print(f"\n=== Final Results (Source: {args.model}, Attack: {args.attack}) ===")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Benign Acc: {data['benign_acc']:.2f}% | ASR: {100 - data['benign_acc']:.2f}%")
        print(f"  Adv Acc: {data['adv_acc']:.2f}% | ASR: {100 - data['adv_acc']:.2f}%")
        print(f"  IKDAdv Acc: {data['fdadv_acc']:.2f}% | ASR: {100 - data['fdadv_acc']:.2f}%")
        print("-" * 50)

    return results



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Attack configurations')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=40)
    parser.add_argument('--attack', type=str, default='mifgsm',
                        choices=['mifgsm', 'difgsm', 'tifgsm', 'nifgsm', 'sinifgsm', 'vmifgsm', 'vnifgsm', 'bsrmifgsm', 'sidmifgsm', 'ggsmifgsm'])
    parser.add_argument(
        '--regularization',
        type=str,
        default='KL',
        choices=['MSE', 'CE', 'KL']
    )
    parser.add_argument('--model', type=str, default='resnet50', choices=list(MODEL_CONFIG.keys()))
    args = parser.parse_args()

    seed_torch(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
    ])

    imagenet_dataset = ImagenetDataset(transform=transform)
    imagenet_loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    result_dir = os.path.join(os.getcwd(), 'results', args.attack, args.model,
                              args.regularization, str(args.seed), str(args.weight))
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, 'log (acc, asr).csv')
    if os.path.exists(out_path):
        os.remove(out_path)

    # 运行测试
    test(args, out_path)
