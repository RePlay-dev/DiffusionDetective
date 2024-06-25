import open_clip
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class OpenClipLinear(nn.Module):
    def __init__(self, num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=False):
        super(OpenClipLinear, self).__init__()
        from huggingface_hub import hf_hub_download
        arch = 'ViT-L-14'
        location = ('laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K', 'open_clip_pytorch_model.bin')
        backbone = open_clip.create_model(arch, pretrained=hf_hub_download(*location))

        self.num_features = backbone.visual.proj.shape[0]
        backbone.visual.proj = None

        self.bb = [backbone, ]
        self.normalize = normalize

        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(OpenClipLinear, self).to(*args, **kwargs)
        return self

    def forward_features(self, x):
        with torch.no_grad():
            self.bb[0].eval()
            features = self.bb[0].encode_image(x, normalize=self.normalize)
        return features

    def forward_head(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class ChannelLinear(nn.Linear):
    def __init__(
            self, in_features: int, out_features: int, bias: bool = True, pool=None
    ) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x):
        axis_ref = len(x.shape) - 1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            stride0=2,
            padding=1,
            dropout=0.0,
            gap_size=None,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=stride0, padding=3 * padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=padding)
        self.layer1 = self._make_layer(block, 64, layers[0], padding=padding)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=padding)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=padding)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=padding)

        if gap_size is None:
            self.gap_size = None
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_size < 0:
            with torch.no_grad():
                y = self.forward_features(
                    torch.zeros((1, 3, -gap_size, -gap_size), dtype=torch.float32)
                ).shape
            print("gap_size:", -gap_size, ">>", y[-1])
            self.gap_size = y[-1]
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        elif gap_size == 1:
            self.gap_size = gap_size
            self.avgpool = None
        else:
            self.gap_size = gap_size
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                padding=padding,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding))

        return nn.Sequential(*layers)

    def change_output(self, num_classes):
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self

    def change_input(self, num_inputs):
        data = self.conv1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs > old_num_inputs:
            times = num_inputs // old_num_inputs
            if (times * old_num_inputs) < num_inputs:
                times = times + 1
            data = data.repeat(1, times, 1, 1) / times
        elif num_inputs == old_num_inputs:
            return self

        data = data[:, :num_inputs, :, :]
        print(self.conv1.weight.data.shape, "->", data.shape)
        self.conv1.weight.data = data

        return self

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x):
        if self.avgpool is not None:
            x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        y = self.fc(x)
        if self.gap_size is None:
            y = torch.squeeze(torch.squeeze(y, -1), -1)
        return y

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]

        out += identity
        out = self.relu(out)

        return out
