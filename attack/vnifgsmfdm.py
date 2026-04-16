import torch
import torch.nn as nn


class LinfFDMVNIFGSMAttack(object):
    def __init__(self, args, model, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, N=5, beta=3 / 2):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
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
        v = torch.zeros_like(images).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.regularization == 'CE':
            regularization = nn.CrossEntropyLoss()
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                nes_images = adv_images + self.decay * self.alpha * momentum
                adv_outputs = self.model(nes_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)

                # Calculate loss
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)

                # Update adversarial images
                adv_grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = (adv_grad + v) / torch.mean(
                    torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
                )
                grad = grad + momentum * self.decay
                momentum = grad

                # Calculate Gradient Variance
                GV_grad = torch.zeros_like(images).detach().to(self.device)
                for _ in range(self.N):
                    neighbor_images = adv_images.detach() + torch.randn_like(
                        images
                    ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                    neighbor_images.requires_grad = True
                    adv_outputs = self.model(neighbor_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)

                    # Calculate loss
                    cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)
                    GV_grad += torch.autograd.grad(
                        cost, neighbor_images, retain_graph=False, create_graph=False
                    )[0]
                # obtaining the gradient variance
                v = GV_grad / self.N - adv_grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self.regularization == 'MSE':
            regularization = nn.MSELoss()
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                nes_images = adv_images + self.decay * self.alpha * momentum
                adv_outputs = self.model(nes_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)

                # Calculate loss
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)

                # Update adversarial images
                adv_grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = (adv_grad + v) / torch.mean(
                    torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
                )
                grad = grad + momentum * self.decay
                momentum = grad

                # Calculate Gradient Variance
                GV_grad = torch.zeros_like(images).detach().to(self.device)
                for _ in range(self.N):
                    neighbor_images = adv_images.detach() + torch.randn_like(
                        images
                    ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                    neighbor_images.requires_grad = True
                    adv_outputs = self.model(neighbor_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)

                    # Calculate loss
                    cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)
                    GV_grad += torch.autograd.grad(
                        cost, neighbor_images, retain_graph=False, create_graph=False
                    )[0]
                # obtaining the gradient variance
                v = GV_grad / self.N - adv_grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self.regularization == 'KL':
            regularization = nn.KLDivLoss(reduction="batchmean")
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                nes_images = adv_images + self.decay * self.alpha * momentum
                adv_outputs = self.model(nes_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                log_adv_outputs = torch.log(adv_outputs)

                # Calculate loss
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, log_adv_outputs)

                # Update adversarial images
                adv_grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = (adv_grad + v) / torch.mean(
                    torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
                )
                grad = grad + momentum * self.decay
                momentum = grad

                # Calculate Gradient Variance
                GV_grad = torch.zeros_like(images).detach().to(self.device)
                for _ in range(self.N):
                    neighbor_images = adv_images.detach() + torch.randn_like(
                        images
                    ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                    neighbor_images.requires_grad = True
                    adv_outputs = self.model(neighbor_images)
                    adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                    log_adv_outputs = torch.log(adv_outputs)

                    # Calculate loss
                    cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, log_adv_outputs)
                    GV_grad += torch.autograd.grad(
                        cost, neighbor_images, retain_graph=False, create_graph=False
                    )[0]
                # obtaining the gradient variance
                v = GV_grad / self.N - adv_grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
