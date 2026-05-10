
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# VP-SDE
# ---------------------------------------------------------------------------

class VPSDE:
    """Variance Preserving SDE: dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw.

    Linear beta schedule: beta(t) = beta_min + t * (beta_max - beta_min).
    """
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_params(self, t):
        """Closed-form mean coefficient and std of q(x_t | x_0)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean_coeff, std

    def perturb(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        mean_coeff, std = self.marginal_params(t)
        mean_coeff = mean_coeff[:, None, None, None]
        std = std[:, None, None, None]
        return mean_coeff * x0 + std * noise, noise


# ---------------------------------------------------------------------------
# Pre-training loss
# ---------------------------------------------------------------------------

def compute_pretrain_loss(net, x0, sde, T=50):
    """VP-SDE denoising loss for unconditional pre-training (Sec.3.1, Eq.3).

    The restoration network (with global residual) takes noisy HDR x_t and
    predicts the clean image x_0. Since net(x_t) = x_t + residual, the
    network naturally learns to denoise by predicting the clean target.

    Args:
        net: restoration network (InverseToneMappingUNet)
        x0:  clean HDR images [B, 3, H, W]
        sde: VPSDE instance
        T:   number of diffusion timesteps (controls max noise level)
    """
    B = x0.shape[0]
    # sample continuous t in [eps, t_max], t_max = T/1000
    t_max = T / 1000.0
    t = torch.rand(B, device=x0.device) * (t_max - 1e-5) + 1e-5

    xt, noise = sde.perturb(x0, t)
    pred = net(xt)
    return F.mse_loss(pred, x0)


# ---------------------------------------------------------------------------
# Parameter Regularization (Eq.4-6)
# ---------------------------------------------------------------------------

class ParameterRegularizer:
    """Importance-weighted regularization to preserve the generative prior.

    Stores pre-trained parameters theta_0 and computes gradient-based
    importance weights Omega (Eq.4-6). The regularization loss:
        L_reg = sum_k [ Omega_k * (theta_k - theta_0_k)^2 ]

    Omega_k is estimated from the Fisher information (squared gradients)
    of the pre-training loss, measuring how important each parameter is
    for the generative prior.
    """
    def __init__(self, net):
        self.theta0 = {}
        self.omega = {}
        for name, p in net.named_parameters():
            self.theta0[name] = p.data.clone()
            self.omega[name] = torch.ones_like(p.data)

    def compute_importance(self, net, dataloader, sde, device, T=50, num_batches=100):
        """Estimate importance weights from pre-training gradients (Eq.4-5).

        Omega_k = E[grad_k^2]  (Fisher information diagonal approximation)
        """
        grad_sq_sum = {n: torch.zeros_like(p) for n, p in net.named_parameters()}
        count = 0

        net.train()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, (list, tuple)):
                x0 = batch[0].to(device)
            else:
                x0 = batch.to(device)

            net.zero_grad()
            loss = compute_pretrain_loss(net, x0, sde, T)
            loss.backward()

            for name, p in net.named_parameters():
                if p.grad is not None:
                    grad_sq_sum[name] += p.grad.data ** 2
            count += 1

        for name in grad_sq_sum:
            self.omega[name] = grad_sq_sum[name] / max(count, 1)

        net.zero_grad()

    def loss(self, net):
        """Compute L_reg = sum_k Omega_k * (theta_k - theta_0_k)^2."""
        reg = torch.tensor(0., device=next(net.parameters()).device)
        for name, p in net.named_parameters():
            delta = p - self.theta0[name].to(p.device)
            reg = reg + (self.omega[name].to(p.device) * delta ** 2).sum()
        return reg


# ---------------------------------------------------------------------------
# Gradient Orthogonal Loss (Eq.7-9)
# ---------------------------------------------------------------------------

class GradientOrthogonalLoss:
    """Aligns generation and restoration gradient directions (Eq.7-9).

    Computes the cosine similarity between the full gradient vectors of
    the generation loss and the restoration loss, then penalises
    misalignment between the two tasks.

    Eq.7: s = <g_gen, g_res>   (cross-task similarity, want s -> 1)
    Eq.8: d = <g_gen, g_gen> + <g_res, g_res>  (intra-task, trivially 1)
    Eq.9: L_orthog = (1 - s) + |d|

    In single-batch training, d is always 1 (self-cosine), so the
    effective loss is (1 - s) + 1. We keep |d| for generality with
    gradient-accumulation settings where per-sample gradients differ.
    """
    @staticmethod
    def compute(net, loss_gen, loss_res, param_reg=None):
        """Compute gradient orthogonal loss from generation and restoration losses.

        Args:
            net: the restoration network
            loss_gen: scalar generation (diffusion) loss
            loss_res: scalar restoration loss
            param_reg: ParameterRegularizer (unused, kept for API compat)
        Returns:
            L_orthog (scalar)
        """
        params = [p for p in net.parameters() if p.requires_grad]

        grads_gen = torch.autograd.grad(loss_gen, params, retain_graph=True,
                                        allow_unused=True)
        grads_res = torch.autograd.grad(loss_res, params, retain_graph=True,
                                        allow_unused=True)

        # flatten all gradients into single vectors
        g_gen_parts = []
        g_res_parts = []
        for g_gen, g_res in zip(grads_gen, grads_res):
            if g_gen is None or g_res is None:
                continue
            g_gen_parts.append(g_gen.flatten())
            g_res_parts.append(g_res.flatten())

        if len(g_gen_parts) == 0:
            return torch.tensor(0., device=params[0].device)

        g_gen_vec = torch.cat(g_gen_parts)
        g_res_vec = torch.cat(g_res_parts)

        # cross-task similarity s (Eq.7)
        s = F.cosine_similarity(g_gen_vec.unsqueeze(0),
                                g_res_vec.unsqueeze(0))

        # L_orthog = (1 - s)  (Eq.9, dropping trivial |d|=1 term)
        return 1.0 - s.squeeze()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from bitm_arch import InverseToneMappingUNet

    device = torch.device('cpu')
    sde = VPSDE()

    # small network for testing
    net = InverseToneMappingUNet(
        img_channel=3, width=16, middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1])
    print(f'params: {sum(p.numel() for p in net.parameters())}')

    # pre-training loss
    x0 = torch.rand(2, 3, 64, 64)
    loss_pt = compute_pretrain_loss(net, x0, sde, T=50)
    print(f'pretrain loss: {loss_pt.item():.4f}')

    # parameter regularizer
    param_reg = ParameterRegularizer(net)
    l_reg = param_reg.loss(net)
    print(f'L_reg (before training, should be ~0): {l_reg.item():.6f}')

    # simulate one gradient step to make params drift
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    opt.zero_grad()
    loss_pt.backward()
    opt.step()

    l_reg_after = param_reg.loss(net)
    print(f'L_reg (after step, should be > 0): {l_reg_after.item():.6f}')

    # gradient orthogonal loss
    x_ldr = torch.rand(2, 3, 64, 64)
    x_hdr = torch.rand(2, 3, 64, 64)

    out_res = net(x_ldr)
    loss_res = F.l1_loss(out_res, x_hdr)

    loss_gen = compute_pretrain_loss(net, x_hdr, sde, T=50)

    l_orthog = GradientOrthogonalLoss.compute(net, loss_gen, loss_res)
    print(f'L_orthog: {l_orthog.item():.4f}')
