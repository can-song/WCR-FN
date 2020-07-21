import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class WCRConv2dFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, stride, padding, alpha):
        ctx.stride = stride
        ctx.padding = padding
        ctx.alpha = alpha
        if bias is None:
            ctx.bias = False
        else:
            ctx.bias = True

        # weight -= weight.mean(dim=-1, keepdim=True).mean(dim=-1, keepdim=True).mean(dim=-1, keepdim=True)
        # weight -= weight.mean(dim=0)
        # weight /= (weight * weight).sum(dim=[1, 2, 3], keepdim=True).sqrt()
        
        out_channels, in_channels, kh, kw = weight.shape
        batch_size, _, height, width = inputs.shape
        kh, kw = weight.shape[2:]
        out_height = (height + padding[0] * 2 - kh) // stride[0] + 1
        out_width = (width + padding[1] * 2 - kw) // stride[1] + 1
        kernel_size = kh, kw

        # x: (batch_size, dim_weight, num_patch)
        unfold_inputs = F.unfold(inputs, kernel_size, 1, padding, stride)
        x = torch.matmul(weight.view(out_channels, -1), unfold_inputs)
        x = x.reshape(batch_size, out_channels, out_height, out_width)
        if ctx.bias:
            x = x + bias.view(-1, 1, 1)
        ctx.save_for_backward(inputs, weight)
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, weight = ctx.saved_variables
        batch_size, out_channels, out_height, out_width = grad_outputs.shape
        _, in_channels, kh, kw = weight.shape
        
        # (batch_size, dim_weight, num_patch)
        unfold_inputs = F.unfold(inputs, (kh, kw), 1, ctx.padding, ctx.stride)

        if ctx.bias:
            grad_bias = grad_outputs.sum(dim=(0, 2, 3))
        else:
            grad_bias = None

        grad = grad_outputs.view(batch_size, out_channels, -1)
        grad_weight = torch.bmm(grad, unfold_inputs.transpose(1, 2)).sum(0).view(-1, in_channels, kh, kw)
        if ctx.alpha is not None:
            grad_weight += weight.sum(dim=0) * ctx.alpha

        grad_inputs = torch.matmul(weight.view(out_channels, -1).transpose(0, 1), grad)
        grad_inputs = F.fold(grad_inputs, inputs.shape[2:],
                            (kh, kw), 1, ctx.padding, ctx.stride)
        return grad_inputs, grad_weight, grad_bias, None, None, None


class WCRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, alpha=0.01):
        super(WCRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias)
        self.alpha = alpha

    def forward(self, inputs):
        # print(id(self.weight))
        outputs = WCRConv2dFunction.apply(inputs, self.weight, self.bias, self.stride, self.padding, self.alpha)
        return outputs


class FeatsNorm(nn.Module):
    def __init__(self):
        super(FeatsNorm, self).__init__()

    def forward(self, x):
        x = F.relu(x, inplace=True)
        x = 0.5 * x.pow(2)
        L = x.mean(dim=1, keepdim=True)
        if L.dtype == torch.float16:
            L += 1e-3
            # L.clamp_min_(1e-3)
        else:
            L += 1e-6
            # L.clamp_min(1e-6)
        y = x / L - 1
        return y


class GroupSaparse(nn.Module):
    def __init__(self, sparsity_factor=4):
        super(GroupSaparse, self).__init__()
        self.sparsity_factor = sparsity_factor

    def forward(self, x):
        x = F.relu(x, inplace=True)
        b, c, h, w = x.size()
        t = x.view(b, self.sparsity_factor, -1, h, w)
        idx = t.argmax(dim=1, keepdim=True)
        mask = torch.zeros_like(t).scatter_(dim=1, index=idx, value=1)
        x = x * mask.view_as(x)
        return x


class Power(nn.Module):
    def __init__(self):
        super(Power, self).__init__()

    def forward(self, x):
        return x.relu().pow(2)


if __name__ == "__main__":
    x = torch.randn((3, 8, 256, 256), requires_grad=True)
    m = WCRConv2d(8, 16, 3, 1, 1, True)
    y = m(x)
    w = torch.randn((16, 8, 3, 3), requires_grad=True)
    b = torch.randn(16, requires_grad=True)
    x1 = torch.tensor(x.detach(), requires_grad=True)
    x2 = torch.tensor(x.detach(), requires_grad=True)
    y1 = WCRConv2dFunction.apply(x1, w, b, 2, 2)
    # x.zero_grad()
    y1.backward(torch.ones_like(y1))
    # dx1 = x1.grad.clone()

    # x2 = x.clone()
    # x2.requres_grad = True
    y2 = F.conv2d(x2, w, b, 2, 2)
    # x.zero_grad()
    y2.backward(torch.ones_like(y2))
    # dx2 = x.grad.clone()
    print((x1.grad-x2.grad).abs().sum())

    print((y1-y2).abs().sum())
    print("debug")