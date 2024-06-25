from importlib import import_module

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from fastai.vision.core import PILImage
from torch import Tensor

from detectors.base import Detector
from utils import device


class DireDetector(Detector):
    def __init__(self, model_path: str):
        self.model = get_network("resnet50")
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)

        self.trans = transforms.Compose(
            (
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            )
        )

        super().__init__(self.model)

    def get_prediction(self, img: PILImage) -> tuple[str, float, int]:
        img_tensor = self.get_processed_tensor(img)
        with torch.no_grad():
            prob = self.model(img_tensor).sigmoid().item()
            # TODO: Calculate correct probability
        return "fake", prob, 0

    def get_processed_tensor(self, img: PILImage) -> Tensor:
        img = self.trans(img.convert("RGB"))

        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        in_tens = img.unsqueeze(0)
        return in_tens.to(device)

    def get_processed_image(self, img: PILImage) -> Tensor:
        img = self.trans(img.convert("RGB"))

        # Permute the tensor dimensions from (3, 224, 224) to (224, 224, 3)
        return img.permute(1, 2, 0)


def get_network(arch: str, is_train=False, continue_train=False, init_gain=0.02, pretrained=True) -> nn.Module:
    if "resnet" in arch:
        from networks.resnet import ResNet

        resnet = getattr(import_module("networks.resnet"), arch)
        if is_train:
            if continue_train:
                model: ResNet = resnet(num_classes=1)
            else:
                model: ResNet = resnet(pretrained=pretrained)
                model.fc = nn.Linear(2048, 1)
                nn.init.normal_(model.fc.weight.data, 0.0, init_gain)
        else:
            model: ResNet = resnet(num_classes=1)
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")
