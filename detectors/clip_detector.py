import torch
import torchvision.transforms as transforms
from fastai.vision.core import PILImage
from torch import Tensor, load
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode

from detectors.base import Detector
from networks.openclipnet import OpenClipLinear
from utils import device, sigmoid


class ClipDetector(Detector):
    def __init__(self):
        model_path = 'models/clipdet_latent10k_plus.pth'

        # create_architecture
        model = OpenClipLinear(num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=True)

        # load_weights
        dat = load(model_path, map_location='cpu')
        model.load_state_dict(dat['model'])
        self.model = model
        super().__init__(self.model)

    def get_prediction(self, img: PILImage) -> tuple[str, float, int]:
        image_tensor = self.get_processed_tensor(img)
        out_tens = self.model(image_tensor).cpu().numpy()
        # For some reason, calling predict moves the model to the CPU
        # We have to ensure the model stays on the correct device so any operations after this stay fast
        self.model.to(device)

        out_tens = out_tens[:, 0]
        logit = out_tens.item(0)

        # Apply sigmoid function to get probability of class "fake"
        probability_fake = sigmoid(logit)

        # Probability of class "real"
        probability_real = 1 - probability_fake

        # Classify based on the probability threshold of 0.5
        prediction = "fake" if probability_fake > 0.5 else "real"

        # Return the prediction and the probability of the predicted class
        if prediction == "fake":
            return prediction, probability_fake, 0
        else:
            return prediction, probability_real, 0

    def get_processed_tensor(self, img: PILImage) -> Tensor:
        processed_image = self.get_processed_image(img)

        # set up tensor normalization
        normalize_transform = list()
        normalize_transform.append(transforms.ToTensor())
        normalize_transform.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
        normalize = Compose(normalize_transform)

        normalized_tensor = normalize(processed_image)
        stacked_tensor = torch.stack([normalized_tensor], 0)  # Adds a dimension to the tensor
        return stacked_tensor.clone().to(device)  # Retrieve the processed image as a tensor

    def get_processed_image(self, img: PILImage) -> PILImage:
        # set up image transforms
        transform = list()
        transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((224, 224)))
        transform = Compose(transform)

        return transform(img)
