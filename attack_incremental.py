import argparse
import os
import random
import shutil
import tempfile

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm

from attack.bsrmifgsm import LinfBSRAttack
from attack.bsrmifgsmfd import LinfFDBSRAttack
from attack.difgsm import LinfDIFGSMAttack
from attack.difgsmfd import LinfFDDIFGSMAttack
from attack.ggsmifgsm import LinfGGSMIFGSMAttack
from attack.ggsmifgsmfd import LinfFDGGSMIFGSMAttack
from attack.mifgsm import LinfMIFGSMAttack
from attack.mifgsmfd import LinfFDMIFGSMAttack
from attack.nifgsm import LinfNIFGSMAttack
from attack.nifgsmfd import LinfFDNIFGSMAttack
from attack.sidmifgsm import LinfSIDMIFGSMAttack
from attack.sidmifgsmfd import LinfFDSIDMIFGSMAttack
from attack.sinifgsm import LinfSINIFGSMAttack
from attack.sinifgsmfd import LinfFDSINIFGSMAttack
from attack.tifgsm import LinfTIFGSMAttack
from attack.tifgsmfd import LinfFDTIFGSMAttack
from attack.vmifgsm import LinfVMIFGSMAttack
from attack.vmifgsmfd import LinfFDVMIFGSMAttack
from attack.vnifgsm import LinfVNIFGSMAttack
from attack.vnifgsmfd import LinfFDVNIFGSMAttack


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dev_data")
IMAGE_DIR = os.path.join(DATA_DIR, "val_rs")
CSV_PATH = os.path.join(DATA_DIR, "val_rs.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------- 模型配置 ----------------------------
MODEL_CONFIG = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    # "resnext50": "resnext50_32x4d",
    # "vgg19bn": "vgg19_bn",
    # "incres_v2": "inception_resnet_v2",
    # "inc_v3": "inception_v3",
    # "inc_v4": "inception_v4",
    # "resnet101": "resnet101",
    # "resnet152": "resnet152",
    # "adv_inception_v3": "adv_inception_v3",
    # "ens_adv_inception_resnet_v2": "ens_adv_inception_resnet_v2",
    # "visformer_small": "visformer_small",
    # "vit_b": "vit_base_patch16_224",
    # "swin_b": "swin_s3_base_224",
    # "pit_b": "pit_b_224",
    # "mobilenet": "mobilenetv2_050",
}


# ---------------------------- 攻击类配置 ----------------------------
ATTACK_CLASSES = {
    "mifgsm": {"base": LinfMIFGSMAttack, "fd": LinfFDMIFGSMAttack},
    "difgsm": {"base": LinfDIFGSMAttack, "fd": LinfFDDIFGSMAttack},
    "tifgsm": {"base": LinfTIFGSMAttack, "fd": LinfFDTIFGSMAttack},
    "nifgsm": {"base": LinfNIFGSMAttack, "fd": LinfFDNIFGSMAttack},
    "sinifgsm": {"base": LinfSINIFGSMAttack, "fd": LinfFDSINIFGSMAttack},
    "vmifgsm": {"base": LinfVMIFGSMAttack, "fd": LinfFDVMIFGSMAttack},
    "vnifgsm": {"base": LinfVNIFGSMAttack, "fd": LinfFDVNIFGSMAttack},
    "bsrmifgsm": {"base": LinfBSRAttack, "fd": LinfFDBSRAttack},
    "sidmifgsm": {"base": LinfSIDMIFGSMAttack, "fd": LinfFDSIDMIFGSMAttack},
    "ggsmifgsm": {"base": LinfGGSMIFGSMAttack, "fd": LinfFDGGSMIFGSMAttack},
}


# ---------------------------- 攻击参数 ----------------------------
COMMON_ATTACK_PARAMS = {
    "eps": 16 / 255,
    "alpha": 2 / 255,
    "steps": 10,
    "decay": 1.0,
}

SPECIFIC_ATTACK_PARAMS = {
    "difgsm": {"resize_rate": 0.9, "diversity_prob": 0.5, "random_start": False},
    "tifgsm": {
        "kernel_name": "gaussian",
        "len_kernel": 15,
        "nsig": 3,
        "resize_rate": 0.9,
        "diversity_prob": 0.5,
        "random_start": False,
    },
    "sinifgsm": {"m": 5},
    "vmifgsm": {"N": 5, "beta": 3 / 2},
    "bsrmifgsm": {"n_blocks": 2, "max_angle": 24, "N": 20},
    "sidmifgsm": {"N": 20, "beta": 0.1, "k": 2, "p": 0.5, "omega": 0.5},
    "ggsmifgsm": {"N": 20, "zeta": 2.0},
}


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, images):
        normalized = (images - self.mean) / self.std
        return self.model(normalized)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def build_normalized_model(model):
    return NormalizedModel(model)


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_model(model_name, device):
    base_model = timm.create_model(MODEL_CONFIG[model_name], pretrained=True)
    normalized_model = build_normalized_model(base_model).to(device)
    return freeze_model(normalized_model)


def create_attackers(args, model, device):
    attack_params = {**COMMON_ATTACK_PARAMS}
    if args.attack in SPECIFIC_ATTACK_PARAMS:
        attack_params.update(SPECIFIC_ATTACK_PARAMS[args.attack])

    attack_cls = ATTACK_CLASSES[args.attack]
    return (
        attack_cls["base"](model=model, device=device, **attack_params),
        attack_cls["fd"](args, model=model, device=device, **attack_params),
    )


def pack_images_for_cache(images):
    return torch.clamp((images.detach().cpu() * 255.0).round(), 0, 255).to(torch.uint8)


def unpack_images_from_cache(images_uint8, device):
    return images_uint8.to(device=device, dtype=torch.float32).div(255.0)


def generate_attack_cache(args, dataloader, device, cache_dir):
    print(f"\nLoading SOURCE model: {args.model} ...")
    source_model = load_model(args.model, device)
    adversary, fdadversary = create_attackers(args, source_model, device)

    cache_paths = []

    for batch_idx, (inputs, targets, img_ids) in enumerate(
        tqdm(dataloader, desc="Generating attack cache")
    ):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.enable_grad():
            adv = adversary.perturb(inputs, targets)
            fdadv = fdadversary.perturb(inputs, targets)

        cache_path = os.path.join(cache_dir, f"batch_{batch_idx:05d}.pt")
        torch.save(
            {
                "inputs": pack_images_for_cache(inputs),
                "adv": pack_images_for_cache(adv),
                "fdadv": pack_images_for_cache(fdadv),
                "targets": targets.detach().cpu(),
                "img_ids": list(img_ids),
            },
            cache_path,
        )
        cache_paths.append(cache_path)

        del inputs, targets, adv, fdadv

    del adversary, fdadversary, source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache_paths


def evaluate_target_model(target_name, cache_paths, total, device):
    print(f"\n========== Testing target model: {target_name} ==========")

    target_model = load_model(target_name, device)

    benign_correct = 0
    adv_correct = 0
    fdadv_correct = 0

    for cache_path in tqdm(cache_paths, desc=f"Evaluating {target_name}"):
        batch = torch.load(cache_path, map_location="cpu")
        inputs = unpack_images_from_cache(batch["inputs"], device)
        adv = unpack_images_from_cache(batch["adv"], device)
        fdadv = unpack_images_from_cache(batch["fdadv"], device)
        targets = batch["targets"].to(device)

        with torch.no_grad():
            benign_outputs = target_model(inputs)
            _, benign_pred = benign_outputs.max(1)
            benign_correct += benign_pred.eq(targets).sum().item()

            adv_outputs = target_model(adv)
            _, adv_pred = adv_outputs.max(1)
            adv_correct += adv_pred.eq(targets).sum().item()

            fdadv_outputs = target_model(fdadv)
            _, fdadv_pred = fdadv_outputs.max(1)
            fdadv_correct += fdadv_pred.eq(targets).sum().item()

        del batch, inputs, adv, fdadv, targets
        del benign_outputs, benign_pred, adv_outputs, adv_pred, fdadv_outputs, fdadv_pred

    result = {
        "benign_acc": 100.0 * benign_correct / total,
        "adv_acc": 100.0 * adv_correct / total,
        "fdadv_acc": 100.0 * fdadv_correct / total,
    }

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Released GPU memory for model: {target_name}")

    return result


def append_result_row(out_path, model_name, data):
    # Historical project convention: ASR here is recorded as misclassification rate.
    line = (
        f"{model_name}_benign_acc: {data['benign_acc']:.4f}, "
        f"{model_name}_benign_asr: {100 - data['benign_acc']:.4f}, "
        f"{model_name}_adv_acc: {data['adv_acc']:.4f}, "
        f"{model_name}_adv_asr: {100 - data['adv_acc']:.4f}, "
        f"{model_name}_fdadv_acc: {data['fdadv_acc']:.4f}, "
        f"{model_name}_fdadv_asr: {100 - data['fdadv_acc']:.4f}\n"
    )

    with open(out_path, "a") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def test(args, dataloader, result_dir, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = len(dataloader.dataset)
    results = {}

    os.makedirs(result_dir, exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    cache_dir = tempfile.mkdtemp(prefix="attack_cache_", dir=result_dir)
    print(f"Cache directory: {cache_dir}")

    try:
        cache_paths = generate_attack_cache(args, dataloader, device, cache_dir)

        for target_name in MODEL_CONFIG:
            result = evaluate_target_model(target_name, cache_paths, total, device)
            results[target_name] = result
            append_result_row(out_path, target_name, result)
            print(f"Saved incremental result for model: {target_name}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Removed cache directory: {cache_dir}")

    print(f"\n=== Final Results (Source: {args.model}, Attack: {args.attack}) ===")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Benign Acc: {data['benign_acc']:.2f}% | ASR: {100 - data['benign_acc']:.2f}%")
        print(f"  Adv Acc: {data['adv_acc']:.2f}% | ASR: {100 - data['adv_acc']:.2f}%")
        print(f"  FDAdv Acc: {data['fdadv_acc']:.2f}% | ASR: {100 - data['fdadv_acc']:.2f}%")
        print("-" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack configurations")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--weight", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument(
        "--attack",
        type=str,
        default="mifgsm",
        choices=[
            "mifgsm",
            "difgsm",
            "tifgsm",
            "nifgsm",
            "sinifgsm",
            "vmifgsm",
            "vnifgsm",
            "bsrmifgsm",
            "sidmifgsm",
            "ggsmifgsm",
        ],
    )
    parser.add_argument(
        "--regularization",
        type=str,
        default="CE",
        choices=["MSE", "CE", "KL"],
    )
    parser.add_argument("--model", type=str, default="resnet50", choices=list(MODEL_CONFIG.keys()))
    args = parser.parse_args()

    seed_torch(args.seed)

    transform = build_transform()
    imagenet_dataset = ImagenetDataset(transform=transform)
    imagenet_loader = torch.utils.data.DataLoader(
        imagenet_dataset, batch_size=args.batch, shuffle=False, num_workers=4
    )

    result_dir = os.path.join(
        RESULTS_DIR, args.attack, args.model, args.regularization, str(args.seed), str(args.weight)
    )
    out_path = os.path.join(result_dir, "log (acc, asr).csv")

    test(args, imagenet_loader, result_dir, out_path)
