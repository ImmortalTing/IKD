import torch
import torch.nn as nn
import torch.nn.functional as F


class LinfFDDIFGSMAttack(object):
    def __init__(self, args, model, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, resize_rate=0.9,
                 diversity_prob=0.5, random_start=False):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.device = device
        self.weight = args.weight
        self.regularization = args.regularization

    def input_diversity(self, x_adv, x):
        # 为了保证对抗样本和干净样本具有相同的变换形式（不同时输入可能导致变换方式不同），因此同时进行输入变换.输入为对抗样本和干净样本
        adv_img_size = x_adv.shape[-1]
        img_size = x.shape[-1]
        adv_img_resize = int(adv_img_size * self.resize_rate)
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            adv_img_size = adv_img_resize
            img_size = img_resize
            adv_img_resize = x_adv.shape[-1]
            img_resize = x.shape[-1]

        adv_rnd = torch.randint(low=adv_img_size, high=adv_img_resize, size=(1,), dtype=torch.int32)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        adv_rescaled = F.interpolate(
            x_adv, size=[adv_rnd, adv_rnd], mode="bilinear", align_corners=False
        )
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        adv_h_rem = adv_img_resize - adv_rnd
        h_rem = img_resize - rnd
        adv_w_rem = adv_img_resize - adv_rnd
        w_rem = img_resize - rnd
        adv_pad_top = torch.randint(low=0, high=adv_h_rem.item(), size=(1,), dtype=torch.int32)
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        adv_pad_bottom = adv_h_rem - adv_pad_top
        pad_bottom = h_rem - pad_top
        adv_pad_left = torch.randint(low=0, high=adv_w_rem.item(), size=(1,), dtype=torch.int32)
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        adv_pad_right = adv_w_rem - adv_pad_left
        pad_right = w_rem - pad_left

        adv_padded = F.pad(
            adv_rescaled,
            [adv_pad_left.item(), adv_pad_right.item(), adv_pad_top.item(), adv_pad_bottom.item()],
            value=0,
        )

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        if torch.rand(1) < self.diversity_prob:
            return adv_padded, padded
        else:
            return x_adv, x
        # return adv_padded, padded if torch.rand(1) < self.diversity_prob else x_adv, x

    def perturb(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        benign_images = images.clone().detach()
        adv_images = images.clone().detach()

        if self.regularization == 'CE':
            regularization = nn.CrossEntropyLoss()
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_outputs, benign_outputs = self.input_diversity(adv_images, benign_images)
                benign_outputs = self.model(benign_outputs)
                benign_outputs = F.softmax(benign_outputs, dim=1)
                adv_logits = self.model(adv_outputs)
                adv_softmax = F.softmax(adv_logits, dim=1)
                cost = loss(adv_softmax, labels) + self.weight * regularization(adv_logits, benign_outputs)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        elif self.regularization == 'MSE':
            regularization = nn.MSELoss()
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_outputs, benign_outputs = self.input_diversity(adv_images, benign_images)
                benign_outputs = self.model(benign_outputs)
                benign_outputs = F.softmax(benign_outputs, dim=1)
                adv_logits = self.model(adv_outputs)
                adv_softmax = F.softmax(adv_logits, dim=1)
                cost = loss(adv_softmax, labels) + self.weight * regularization(adv_logits, benign_outputs)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        elif self.regularization == 'KL':
            regularization = nn.KLDivLoss(reduction='batchmean')
            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_outputs, benign_outputs = self.input_diversity(adv_images, benign_images)
                benign_outputs = self.model(benign_outputs)
                benign_outputs = F.softmax(benign_outputs, dim=1)
                adv_logits = self.model(adv_outputs)
                adv_softmax = F.softmax(adv_logits, dim=1)
                log_adv_outputs = F.log_softmax(adv_logits, dim=1)
                cost = loss(adv_softmax, labels) + self.weight * regularization(log_adv_outputs, benign_outputs)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
