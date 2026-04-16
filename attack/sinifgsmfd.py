import torch
import torch.nn as nn


class LinfFDSINIFGSMAttack(object):

    def __init__(self, args, model, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, m=5):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.device = device
        self.weight = args.weight
        self.regularization = args.regularization

    def perturb(self, images, labels):
        r"""
        Overridden.
        """

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
                adv_images.requires_grad = True
                nes_image = adv_images + self.decay * self.alpha * momentum
                # Calculate sum the gradients over the scale copies of the input image
                adv_grad = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    nes_images = nes_image / torch.pow(2, i)
                    adv_outputs = self.model(nes_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    # Calculate loss
                    cost = loss(adv_outputs, labels) + self.weight * regularization(adv_outputs, benign_outputs)
                    adv_grad += torch.autograd.grad(
                        cost, adv_images, retain_graph=False, create_graph=False
                    )[0]
                adv_grad = adv_grad / self.m

                # Update adversarial images
                grad = self.decay * momentum + adv_grad / torch.mean(
                    torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
                )
                momentum = grad
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self.regularization == 'MSE':
            regularization = nn.MSELoss()
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                nes_image = adv_images + self.decay * self.alpha * momentum
                # Calculate sum the gradients over the scale copies of the input image
                adv_grad = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    nes_images = nes_image / torch.pow(2, i)
                    adv_outputs = self.model(nes_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    # Calculate loss
                    cost = loss(adv_outputs, labels) + self.weight * regularization(adv_outputs, benign_outputs)
                    adv_grad += torch.autograd.grad(
                        cost, adv_images, retain_graph=False, create_graph=False
                    )[0]
                adv_grad = adv_grad / self.m

                # Update adversarial images
                grad = self.decay * momentum + adv_grad / torch.mean(
                    torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
                )
                momentum = grad
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self.regularization == 'KL':
            regularization = nn.KLDivLoss(reduction="batchmean")
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                nes_image = adv_images + self.decay * self.alpha * momentum
                # Calculate sum the gradients over the scale copies of the input image
                adv_grad = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    nes_images = nes_image / torch.pow(2, i)
                    adv_outputs = self.model(nes_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    log_adv_outputs = torch.log(adv_outputs)
                    # Calculate loss
                    cost = loss(adv_outputs, labels) + self.weight * regularization(log_adv_outputs, benign_outputs)
                    adv_grad += torch.autograd.grad(
                        cost, adv_images, retain_graph=False, create_graph=False
                    )[0]
                adv_grad = adv_grad / self.m

                # Update adversarial images
                grad = self.decay * momentum + adv_grad / torch.mean(
                    torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
                )
                momentum = grad
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
