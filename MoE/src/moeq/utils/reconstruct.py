import torch
import torch.nn.functional as F

@torch.no_grad()
def error_computing(quantized_matrix, origin_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    print(origin_matrix.shape, quantized_matrix.shape, mse)
    return mse

@torch.no_grad()
def kl_div(quantized_matrix, origin_matrix):
    tensor1 = F.softmax(quantized_matrix, dim=-1)
    tensor2 = F.softmax(origin_matrix, dim=-1)

    tensor1 = tensor1.clamp(min=1e-6)
    tensor2 = tensor2.clamp(min=1e-6)
    kl_div = F.kl_div(torch.log(tensor2), tensor1, reduction='batchmean')
    return kl_div

@torch.no_grad()
def ssim(x, y, C1=0.01**2, C2=0.03**2):
    n, m = x.shape

    mu_x = x.mean(dim=1, keepdim=True)
    mu_y = y.mean(dim=1, keepdim=True)

    sigma_x = x.var(dim=1, unbiased=False, keepdim=True)
    sigma_y = y.var(dim=1, unbiased=False, keepdim=True)

    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=1, keepdim=True)

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
    ssim_vals = numerator / denominator

    return ssim_vals.mean()

def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')

