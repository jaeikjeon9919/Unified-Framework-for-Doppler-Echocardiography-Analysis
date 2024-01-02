import os
import typing as T
import antialiased_cnns
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

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


class UNet(nn.Module):
    def __init__(self, num_classes, antialiasing, context_embedding, in_channels=1):
        super(UNet, self).__init__()
        nb_filter = [16, 32, 64, 128, 256]
        self.context_embedding = context_embedding

        self.enc1_1 = ConvBNReLU(in_channels, nb_filter[0])
        self.enc1_2 = ConvBNReLU(nb_filter[0], nb_filter[0])
        if antialiasing:
            self.pool1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1), antialiased_cnns.BlurPool(nb_filter[0], stride=2),
            )
        else:
            self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBNReLU(nb_filter[0], nb_filter[1])
        self.enc2_2 = ConvBNReLU(nb_filter[1], nb_filter[1])
        if antialiasing:
            self.pool2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1), antialiased_cnns.BlurPool(nb_filter[1], stride=2),
            )
        else:
            self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBNReLU(nb_filter[1], nb_filter[2])
        self.enc3_2 = ConvBNReLU(nb_filter[2], nb_filter[2])
        if antialiasing:
            self.pool3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1), antialiased_cnns.BlurPool(nb_filter[2], stride=2),
            )
        else:
            self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = ConvBNReLU(nb_filter[2], nb_filter[3])
        self.enc4_2 = ConvBNReLU(nb_filter[3], nb_filter[3])
        if antialiasing:
            self.pool4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1), antialiased_cnns.BlurPool(nb_filter[3], stride=2),
            )
        else:
            self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = ConvBNReLU(nb_filter[3], nb_filter[4])
        self.ceblock = CEBlock(nb_filter[3], nb_filter[3])

        self.dec5_1 = ConvBNReLU(nb_filter[4], nb_filter[3])

        self.unpool4 = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = ConvBNReLU(2 * nb_filter[3], nb_filter[3])
        self.dec4_1 = ConvBNReLU(nb_filter[3], nb_filter[2])

        self.unpool3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = ConvBNReLU(2 * nb_filter[2], nb_filter[2])
        self.dec3_1 = ConvBNReLU(nb_filter[2], nb_filter[1])

        self.unpool2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = ConvBNReLU(2 * nb_filter[1], nb_filter[1])
        self.dec2_1 = ConvBNReLU(nb_filter[1], nb_filter[0])

        self.unpool1 = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = ConvBNReLU(2 * nb_filter[0], nb_filter[0])
        self.dec1_1 = ConvBNReLU(nb_filter[0], nb_filter[0])

        self.fc = nn.Conv2d(nb_filter[0], out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)
        if self.context_embedding:
            dec5_1, doppler_shape_pred = self.ceblock(dec5_1)
        else:
            doppler_shape_pred = None

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)
        return output, doppler_shape_pred

    def inference_forward(self, x):
        logits, _ = self.forward(x)
        return logits


class UNetWrapper(nn.Module):
    def __init__(self, num_classes, antialiasing, context_embedding, deep_supervision=False):
        super(UNetWrapper, self).__init__()
        self.model = UNet(num_classes, antialiasing=antialiasing, context_embedding=context_embedding)

        self.CELoss = nn.CrossEntropyLoss()
        self.contextCELoss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input_batch: T.Dict) -> T.Dict:
        x = input_batch["input"]
        x, doppler_shape_pred = self.model(x)
        losses = self.loss(x, input_batch["label"].long(), doppler_shape_pred, input_batch["label_context"])
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

    def loss(self, x, label, doppler_shape_pred, context_label):
        losses = {}
        losses["seg_loss"] = self.CELoss(x, label)

        if doppler_shape_pred is not None:
            losses["task_loss"] = self.contextCELoss(doppler_shape_pred, context_label)
        else:
            pass
        return losses


if __name__ == "__main__":
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    module = True
    model = UNetWrapper(2, antialiasing=module, context_embedding=module).cuda()
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
