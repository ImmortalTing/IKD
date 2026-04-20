import torch
import torch.nn as nn
import torch.nn.functional as F


class LinfFDGGSMIFGSMAttack(object):

    def __init__(self, args, model, device,
                 eps=16/255, alpha=2/255, steps=10, decay=1.0,
                 N=20, zeta=2.0):

        self.model = model
        self.device = device

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

        self.N = N
        self.zeta = zeta

        self.weight = args.weight
        self.regularization = args.regularization

    def perturb(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).to(self.device)

        adv_images = images.clone().detach()

        CE = nn.CrossEntropyLoss()

        # clean prediction
        with torch.no_grad():
            clean_logits = self.model(images)
            clean_prob = F.softmax(clean_logits, dim=1)

        if self.regularization == "MSE":
            reg = nn.MSELoss()

        elif self.regularization == "CE":
            reg = nn.CrossEntropyLoss()

        elif self.regularization == "KL":
            reg = nn.KLDivLoss(reduction="batchmean")

        for step in range(self.steps):

            prev_grad = torch.zeros_like(images).to(self.device)

            grad_list = []

            # ---------- GGS sampling ----------
            for i in range(self.N):

                noise = torch.empty_like(images).uniform_(
                    -self.zeta*self.eps,
                    self.zeta*self.eps
                )

                if i == 0:
                    direction = torch.sign(noise)
                else:
                    direction = torch.sign(prev_grad)

                x_sample = adv_images + torch.abs(noise)*direction

                x_sample = x_sample.detach()
                x_sample.requires_grad = True

                logits = self.model(x_sample)

                prob = F.softmax(logits, dim=1)

                # ---------- FD loss ----------
                attack_loss = CE(prob, labels)

                if self.regularization == "KL":

                    log_prob = F.log_softmax(logits, dim=1)

                    reg_loss = reg(log_prob, clean_prob)

                elif self.regularization == "CE":

                    reg_loss = reg(logits, clean_prob)

                else:

                    reg_loss = reg(logits, clean_prob)

                loss = attack_loss + self.weight * reg_loss

                grad = torch.autograd.grad(
                    loss,
                    x_sample,
                    retain_graph=False,
                    create_graph=False
                )[0]

                prev_grad = grad.detach()

                grad_list.append(grad)

            # ---------- gradient aggregation ----------
            grad = torch.stack(grad_list).mean(dim=0)

            grad = grad / torch.mean(
                torch.abs(grad),
                dim=(1,2,3),
                keepdim=True
            )

            momentum = momentum*self.decay + grad

            adv_images = adv_images + self.alpha*momentum.sign()

            delta = torch.clamp(
                adv_images-images,
                min=-self.eps,
                max=self.eps
            )

            adv_images = torch.clamp(
                images+delta,
                min=0,
                max=1
            ).detach()

        return adv_images