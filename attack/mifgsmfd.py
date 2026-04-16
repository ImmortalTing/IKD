import torch
import torch.nn as nn
import torch.nn.functional as F


class LinfFDMIFGSMAttack(object):
    def __init__(self, args, model, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.device = device
        self.weight = args.weight
        self.regularization = args.regularization

    def _normalize_and_update(self, grad, momentum, adv_images, images):
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * self.decay
        momentum = grad

        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images, momentum

    def perturb(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.regularization == 'CE':
            regularization = nn.CrossEntropyLoss()
            benign_logits = self.model(images)
            benign_outputs = F.softmax(benign_logits, dim=1)

            try:
                _ = regularization(benign_logits, benign_outputs)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Current PyTorch build does not support soft-target CrossEntropyLoss for "
                    "regularization='CE'. Please use KL or upgrade PyTorch."
                ) from exc

            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_logits = self.model(adv_images)

                adv_softmax = F.softmax(adv_logits, dim=1)
                
                # cls_loss = loss(adv_logits, labels)
                cls_loss = loss(adv_softmax, labels)
                reg_loss = regularization(adv_logits, benign_outputs)
                # reg_loss = regularization(adv_softmax, benign_outputs)
                cost = cls_loss + self.weight * reg_loss

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                adv_images, momentum = self._normalize_and_update(grad, momentum, adv_images, images)

        elif self.regularization == 'MSE':
            regularization = nn.MSELoss()
            benign_logits = self.model(images)
            benign_outputs = F.softmax(benign_logits, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_logits = self.model(adv_images)
                adv_softmax = F.softmax(adv_logits, dim=1)
                cls_loss = loss(adv_softmax, labels)
                # reg_loss = regularization(adv_softmax, benign_outputs)
                reg_loss = regularization(adv_logits, benign_outputs)
                cost = cls_loss + self.weight * reg_loss

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                adv_images, momentum = self._normalize_and_update(grad, momentum, adv_images, images)

        elif self.regularization == 'KL':
            regularization = nn.KLDivLoss(reduction="batchmean")
            benign_logits = self.model(images)
            benign_outputs = F.softmax(benign_logits, dim=1)

            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_logits = self.model(adv_images)

                adv_softmax = F.softmax(adv_logits, dim=1)

                log_adv_outputs = F.log_softmax(adv_logits, dim=1)
                # cls_loss = loss(adv_logits, labels)
                cls_loss = loss(adv_softmax, labels)
                reg_loss = regularization(log_adv_outputs, benign_outputs)
                cost = cls_loss + self.weight * reg_loss

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                adv_images, momentum = self._normalize_and_update(grad, momentum, adv_images, images)
        else:
            raise ValueError(
                "Unsupported regularization "
                f"{self.regularization!r}. Expected one of: 'MSE', 'CE', 'KL'."
            )
        # for _ in range(self.steps):
        #     adv_images.requires_grad = True
        #     benign_outputs = self.model(images)
        #     benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
        #     adv_outputs = self.model(adv_images)
        #     adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
        #     cost = loss(adv_outputs, labels) + self.weight * loss(adv_outputs, benign_outputs)
        #
        #     grad = torch.autograd.grad(
        #         cost, adv_images, retain_graph=False, create_graph=False
        #     )[0]
        #
        #     grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        #     grad = grad + momentum * self.decay
        #     momentum = grad
        #
        #     adv_images = adv_images.detach() + self.alpha * grad.sign()
        #     delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        #     adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
