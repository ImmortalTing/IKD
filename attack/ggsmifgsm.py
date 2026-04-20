import torch
import torch.nn as nn
import torch.nn.functional as F


class LinfGGSMIFGSMAttack:
    """
    GGS + MI-FGSM
    """

    def __init__(
        self,
        model,
        device,
        eps=16/255,
        alpha=2/255,
        steps=10,
        decay=1.0,
        N=20,
        zeta=2.0
    ):
        self.model = model
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        # GGS parameters
        self.N = N
        self.zeta = zeta

        self.loss = nn.CrossEntropyLoss()

    def perturb(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        momentum = torch.zeros_like(images).to(self.device)

        for step in range(self.steps):

            grad_list = []
            prev_grad = torch.zeros_like(images).to(self.device)

            # --------------------------
            # GGS inner sampling
            # --------------------------
            for i in range(self.N):

                noise = torch.empty_like(images).uniform_(
                    -self.zeta * self.eps,
                    self.zeta * self.eps
                )

                if i == 0:
                    direction = torch.sign(noise)
                else:
                    direction = torch.sign(prev_grad)

                x_sample = adv_images + torch.abs(noise) * direction
                x_sample = x_sample.detach()
                x_sample.requires_grad = True

                outputs = self.model(x_sample)
                outputs = F.softmax(outputs, dim=1)

                loss = self.loss(outputs, labels)

                grad = torch.autograd.grad(
                    loss,
                    x_sample,
                    retain_graph=False,
                    create_graph=False
                )[0]

                prev_grad = grad.detach()

                grad_list.append(grad)

            # --------------------------
            # gradient aggregation
            # --------------------------
            grad = torch.stack(grad_list).mean(dim=0)

            grad_norm = torch.mean(
                torch.abs(grad),
                dim=(1, 2, 3),
                keepdim=True
            )

            grad = grad / (grad_norm + 1e-8)

            momentum = self.decay * momentum + grad

            adv_images = adv_images + self.alpha * torch.sign(momentum)

            delta = torch.clamp(
                adv_images - images,
                min=-self.eps,
                max=self.eps
            )

            adv_images = torch.clamp(
                images + delta,
                min=0,
                max=1
            ).detach()

        return adv_images