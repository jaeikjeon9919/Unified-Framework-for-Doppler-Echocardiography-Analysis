import os
import typing as T
import antialiased_cnns
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["NestedUNet"]


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
        self.conv_last = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        doppler_shape_pred = self.context_head(feat)[:, :, 0, 0]
        feat = feat + x
        feat = self.conv_last(feat)
        return feat, doppler_shape_pred


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class NestedUNet(nn.Module):
    def __init__(
        self, num_classes, in_channels=1, deep_supervision=True, antialiasing=False, context_embedding=False,
    ):
        super().__init__()
        nb_filter = [16, 32, 64, 128, 256]
        self.context_embedding = context_embedding
        self.deep_supervision = deep_supervision

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        if antialiasing:
            self.pool1 = nn.Sequential(nn.MaxPool2d(2, 1), antialiased_cnns.BlurPool(nb_filter[0], stride=2))
        else:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        if antialiasing:
            self.pool2 = nn.Sequential(nn.MaxPool2d(2, 1), antialiased_cnns.BlurPool(nb_filter[1], stride=2))
        else:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        if antialiasing:
            self.pool3 = nn.Sequential(nn.MaxPool2d(2, 1), antialiased_cnns.BlurPool(nb_filter[2], stride=2))
        else:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        if antialiasing:
            self.pool4 = nn.Sequential(nn.MaxPool2d(2, 1), antialiased_cnns.BlurPool(nb_filter[3], stride=2))
        else:
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ceblock = CEBlock(nb_filter[4], nb_filter[4])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.fc = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        pool1 = self.pool1(x0_0)

        x1_0 = self.conv1_0(pool1)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        pool2 = self.pool2(x1_0)

        x2_0 = self.conv2_0(pool2)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        pool3 = self.pool3(x2_0)

        x3_0 = self.conv3_0(pool3)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        pool4 = self.pool4(x3_0)

        x4_0 = self.conv4_0(pool4)
        if self.context_embedding:
            x4_0, doppler_shape_pred = self.ceblock(x4_0)
        else:
            doppler_shape_pred = None

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.fc(x0_1)
            output2 = self.fc(x0_2)
            output3 = self.fc(x0_3)
            output4 = self.fc(x0_4)
            return (output1 + output2 + output3 + output4) / 4, doppler_shape_pred
        else:
            output = self.fc(x0_4)
            return output, doppler_shape_pred

    def inference_forward(self, x):
        logits, _ = self.forward(x)
        return logits


class NestedUNetWrapper(nn.Module):
    def __init__(self, num_classes, deep_supervision=True, antialiasing=False, context_embedding=False):
        super(NestedUNetWrapper, self).__init__()
        self.model = NestedUNet(
            num_classes,
            deep_supervision=deep_supervision,
            antialiasing=antialiasing,
            context_embedding=context_embedding,
        )

        self.CELoss = nn.CrossEntropyLoss()
        self.contextCELoss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input_batch: T.Dict) -> T.Dict:
        x = input_batch["input"]
        x, doppler_shape_pred = self.model(x)
        losses = self.loss(x, doppler_shape_pred, input_batch["label"].long(), input_batch["label_context"])
        return {"prediction": x, "loss": sum(losses.values())}

    def inference_forward(self, input_batch: T.Dict):
        output = self.forward(input_batch)
        loss = self.CELoss(output["prediction"], input_batch["label"].long())
        return {"prediction": F.log_softmax(output["prediction"], dim=1), "loss": loss}

    def predict(self, x):
        x, doppler_shape_pred = self.model(x["input"])
        return F.log_softmax(x, dim=1)

    def _forward(self, x):
        output = self.model.inference_forward(x)
        return output

    def loss(self, x, doppler_shape_pred, label, context_label):
        losses = {}
        losses["seg_loss"] = self.CELoss(x, label)

        if doppler_shape_pred is not None:
            losses["task_loss"] = self.contextCELoss(doppler_shape_pred, context_label)
        else:
            pass
        return losses


if __name__ == "__main__":
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    module = True
    model = NestedUNetWrapper(2, antialiasing=module, context_embedding=module).cuda()
    model.eval()

    x = torch.zeros((4, 1, 256, 512)).cuda()
    label = torch.zeros(4, 256, 512).cuda()
    label_context = torch.randint(0, 8, (4, 1))[:, 0].cuda()
    input_batch = {"input": x, "label": label, "label_context": label_context}

    s = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = model(input_batch)
    optimizer.zero_grad()
    output["loss"].backward()
    print(time.time() - s)

    s = time.time()
    with torch.no_grad():
        out = model.inference_forward(input_batch)
    print(time.time() - s)
