import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
import antialiased_cnns

backbone_url = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth"


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, inplace=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class CEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_gap = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.context_head = nn.Conv2d(in_channels, out_channels=9, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_last = ConvBNReLU(out_channels, out_channels, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        doppler_shape_pred = self.context_head(feat)[:, :, 0, 0]
        feat = feat + x
        feat = self.conv_last(feat)
        return feat, doppler_shape_pred


class UpSample(nn.Module):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)


class DetailBranch(nn.Module):
    def __init__(self, antialiasing):
        super(DetailBranch, self).__init__()
        if antialiasing:
            self.S1 = nn.Sequential(
                ConvBNReLU(1, 64, 3, stride=1),
                antialiased_cnns.BlurPool(64, stride=2),
                ConvBNReLU(64, 64, 3, stride=1),
            )
            self.S2 = nn.Sequential(
                ConvBNReLU(64, 64, 3, stride=1),
                antialiased_cnns.BlurPool(64, stride=2),
                ConvBNReLU(64, 64, 3, stride=1),
                ConvBNReLU(64, 64, 3, stride=1),
            )
            self.S3 = nn.Sequential(
                ConvBNReLU(64, 128, 3, stride=1),
                antialiased_cnns.BlurPool(128, stride=2),
                ConvBNReLU(128, 128, 3, stride=1),
                ConvBNReLU(128, 128, 3, stride=1),
            )
        else:
            self.S1 = nn.Sequential(ConvBNReLU(1, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1))
            self.S2 = nn.Sequential(
                ConvBNReLU(64, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1), ConvBNReLU(64, 64, 3, stride=1),
            )
            self.S3 = nn.Sequential(
                ConvBNReLU(64, 128, 3, stride=2), ConvBNReLU(128, 128, 3, stride=1), ConvBNReLU(128, 128, 3, stride=1),
            )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):
    def __init__(self, antialiasing):
        super(StemBlock, self).__init__()
        if antialiasing:
            self.conv = nn.Sequential(ConvBNReLU(1, 16, 3, stride=1), antialiased_cnns.BlurPool(16, stride=2),)
            self.left = nn.Sequential(
                ConvBNReLU(16, 8, 1, stride=1, padding=0),
                ConvBNReLU(8, 16, 3, stride=1),
                antialiased_cnns.BlurPool(16, stride=2),
            )
            self.right = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
                antialiased_cnns.BlurPool(16, stride=2),
            )
        else:
            self.conv = ConvBNReLU(1, 16, 3, stride=2)
            self.left = nn.Sequential(ConvBNReLU(16, 8, 1, stride=1, padding=0), ConvBNReLU(8, 16, 3, stride=2),)
            self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False,),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6, antialiasing=False):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)

        if antialiasing:
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False,),
                nn.BatchNorm2d(mid_chan),
                antialiased_cnns.BlurPool(mid_chan, stride=2),
            )
        else:
            self.dwconv1 = nn.Sequential(
                nn.Conv2d(in_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False,),
                nn.BatchNorm2d(mid_chan),
            )

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False,),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True

        if antialiasing:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, groups=in_chan, bias=False,),
                nn.BatchNorm2d(in_chan),
                antialiased_cnns.BlurPool(in_chan, stride=2),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False,),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):
    def __init__(self, antialiasing=False):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock(antialiasing)
        self.S3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32),)
        self.S4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64),)
        self.S5_4 = nn.Sequential(GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128))
        self.S5_5 = CEBlock(in_channels=128, out_channels=128)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5, doppler_shape_pred = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5, doppler_shape_pred


class BGALayer(nn.Module):
    def __init__(self, antialiasing, f_size1=128, f_size2=128, f_size3=128):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(f_size1, f_size1, kernel_size=3, stride=1, padding=1, groups=f_size1, bias=False),
            nn.BatchNorm2d(f_size1),
            nn.Conv2d(f_size1, f_size1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        if antialiasing:
            self.left2 = nn.Sequential(
                nn.Conv2d(f_size1, f_size1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(f_size1),
                antialiased_cnns.BlurPool(f_size1, stride=2),
                antialiased_cnns.BlurPool(f_size1, stride=2),
            )
        else:
            self.left2 = nn.Sequential(
                nn.Conv2d(f_size1, f_size1, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(f_size1),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            )
        self.right1 = nn.Sequential(
            nn.Conv2d(f_size2, f_size2, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(f_size2),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(f_size2, f_size2, kernel_size=3, stride=1, padding=1, groups=f_size2, bias=False),
            nn.BatchNorm2d(f_size2),
            nn.Conv2d(f_size2, f_size2, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        # TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(f_size3, f_size3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f_size3),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)

        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)

        out = self.conv(left + right)
        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(nn.Upsample(scale_factor=2), ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),)
            if aux
            else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Module):
    def __init__(self, n_classes, antialiasing, context_embedding, aux_mode="train"):
        super(BiSeNetV2, self).__init__()

        self.context_embedding = context_embedding
        self.aux_mode = aux_mode
        self.detail = DetailBranch(antialiasing)
        self.segment = SegmentBranch(antialiasing)
        self.bga = BGALayer(antialiasing)

        # TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)

        if self.aux_mode == "train":
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)

        self.init_weights()

    def forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s, doppler_shape_pred = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        doppler_shape_pred = doppler_shape_pred if self.context_embedding else None

        logits = self.head(feat_head)
        logits_aux2 = self.aux2(feat2)
        logits_aux3 = self.aux3(feat3)
        logits_aux4 = self.aux4(feat4)
        logits_aux5_4 = self.aux5_4(feat5_4)
        return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4, doppler_shape_pred

    def inference_forward(self, x):
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s, doppler_shape_pred = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        logits = self.head(feat_head)
        return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if not module.bias is None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, "last_bn") and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if "head" in name or "aux" in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class BiSeNetV2Wrapper(nn.Module):
    def __init__(self, num_classes, antialiasing, context_embedding):
        super(BiSeNetV2Wrapper, self).__init__()
        self.model = BiSeNetV2(num_classes, antialiasing=antialiasing, context_embedding=context_embedding)

        self.criteria_pre = OhemCELoss(0.7)
        self.criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # 4 for number of auxiliary heads

        self.CELoss = nn.CrossEntropyLoss(ignore_index=255)
        self.contextCELoss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input_batch):
        input = input_batch["input"]
        logits, *logits_aux, doppler_shape_pred = self.model(input)
        loss = self.loss(
            logits, logits_aux, doppler_shape_pred, input_batch["label"].long(), input_batch["label_context"]
        )
        return {"prediction": logits, "loss": loss}

    def inference_forward(self, input_batch):
        input = input_batch["input"]
        output = self.model.inference_forward(input)
        loss = self.CELoss(output, input_batch["label"].long())
        return {"prediction": F.log_softmax(output, dim=1), "loss": loss}

    def _forward(self, x):
        output = self.model.inference_forward(x)
        return output

    def loss(self, logits, logits_aux, doppler_shape_pred, label, context_label):
        loss_pre = self.criteria_pre(logits, label)
        loss_aux = [crit(lgt, label) for crit, lgt in zip(self.criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)

        losses = {}
        losses["seg_loss"] = loss
        if doppler_shape_pred is not None:
            losses["task_loss"] = self.contextCELoss(doppler_shape_pred, context_label)
        return sum(losses.values())


if __name__ == "__main__":
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    module = False
    model = BiSeNetV2Wrapper(2, antialiasing=module, context_embedding=module).cuda()
    model.eval()

    x = torch.zeros((4, 1, 256, 512)).cuda()
    label = torch.zeros(4, 256, 512).cuda()
    label_context = torch.randint(0, 8, (4, 1))[:, 0].cuda()
    input_batch = {"input": x, "label": label, "label_context": label_context}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    s = time.time()
    output = model(input_batch)
    optimizer.zero_grad()
    output["loss"].backward()
    print("update :", time.time() - s)

    s = time.time()
    with torch.no_grad():
        out = model.inference_forward(input_batch)
    print("inference :", time.time() - s)
