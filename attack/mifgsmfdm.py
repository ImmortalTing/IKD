import torch
import torch.nn as nn


class LinfFDMMIFGSMAttack(object):
    def __init__(self, args, model, device, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.device = device
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
                adv_images.requires_grad = True
                adv_outputs = self.model(adv_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)

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
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_outputs = self.model(adv_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, adv_outputs)

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
            regularization = nn.KLDivLoss(reduction="batchmean")
            benign_outputs = self.model(images)
            benign_outputs = nn.functional.softmax(benign_outputs, dim=1)
            for _ in range(self.steps):
                adv_images.requires_grad = True
                adv_outputs = self.model(adv_images)
                adv_outputs = nn.functional.softmax(adv_outputs, dim=1)
                log_adv_outputs = torch.log(adv_outputs)
                cost = loss(adv_outputs, labels) - self.weight * regularization(benign_outputs, log_adv_outputs)

                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * self.decay
                momentum = grad

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
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
