import torch
import torch.nn as nn
import torch.nn.functional as F
from .local_arch import Local_Base


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """SGM (Eq.12): split channels in half, element-wise multiply."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFBlock with SGM and SCA (Figure 5)."""
    def __init__(self, c, kernel_size=3, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_ch = c * DW_Expand

        # spatial interaction
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size, padding=(kernel_size-1)//2, groups=dw_ch)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        # SCA (Eq.14)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.sg = SimpleGate()

        # channel interaction (FFN)
        ffn_ch = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class InverseToneMappingUNet(nn.Module):
    """Efficient U-Net for Inverse Tone Mapping (Figure 5).

    NAFNet backbone with SGM + SCA blocks and global residual connection.
    HDR_output = LDR_input + network_residual
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 drop_flag=False, drop_rate=0.4, kernel_size=3):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.drop_flag = drop_flag
        if drop_flag:
            self.dropout = nn.Dropout2d(p=drop_rate)

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, kernel_size) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, kernel_size) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, kernel_size) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        base = inp[:, :3, :, :]

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        if self.drop_flag:
            x = self.dropout(x)

        x = self.ending(x)
        x = x + base

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class InverseToneMappingUNetLocal(Local_Base, InverseToneMappingUNet):
    def __init__(self, *args, patch_size=512, factor=1.5, fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        InverseToneMappingUNet.__init__(self, *args, **kwargs)
        train_size = (1, 3, patch_size, patch_size)
        N, C, H, W = train_size
        base_size = (int(H * factor), int(W * factor))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 6
    dec_blks = [1, 1, 1, 1]

    net = InverseToneMappingUNet(
        img_channel=3, width=32, middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, kernel_size=3)

    x = torch.randn([1, 3, 256, 256])
    out = net(x)
    print('output size:', out.size())
    print('#parameters:', sum(p.numel() for p in net.parameters()))
