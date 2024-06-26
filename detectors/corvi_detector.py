import torch
import torchvision.transforms as transforms
from fastai.vision.core import PILImage
from torch import Tensor, load
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop

from detectors.base import Detector
from networks.openclipnet import resnet50
from utils import device, sigmoid


class CorviDetector(Detector):
    def __init__(self):
        model_path = 'models/corvi2023.pth'

        # create_architecture
        model = resnet50(num_classes=1, stride0=1, dropout=0.5)

        # load_weights
        dat = load(model_path, map_location='cpu')
        model.load_state_dict(dat['model'])
        self.model = model
        super().__init__(self.model)

    def get_prediction(self, img: PILImage) -> tuple[str, float, int]:
        image_tensor = self.get_processed_tensor(img)
        out_tens = self.model(image_tensor).cpu().numpy()
        out_tens = out_tens[:, 0]
        logit = out_tens.item(0)

        # Apply sigmoid function to get probability of class "fake"
        probability_fake = sigmoid(logit)

        # Probability of class "real"
        probability_real = 1 - probability_fake

        # Classify based on the probability threshold of 0.5
        prediction = "fake" if probability_fake > 0.5 else "real"

        print(f'Corvi detected the image as {prediction}')
        # Return the prediction and the probability of the predicted class
        if prediction == "fake":
            return prediction, probability_fake, 0
        else:
            return prediction, probability_real, 0

    def get_processed_tensor(self, img: PILImage) -> Tensor:
        img = self.get_processed_image(img)

        # set up tensor normalization
        normalize_transform = list()
        normalize_transform.append(transforms.ToTensor())
        normalize_transform.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        normalize = Compose(normalize_transform)

        normalized_tensor = normalize(img)
        stacked_tensor = torch.stack([normalized_tensor], 0)  # Adds a dimension to the tensor
        return stacked_tensor.clone().to(device)  # Retrieve the processed image as a tensor

    def get_processed_image(self, img: PILImage) -> PILImage:
        # While they don't have resize logic in the original Corvi test code they resize and crop the test data to 200x200 beforehand.
        # set up image transforms
        transform = list()
        transform.append(Resize(200, interpolation=InterpolationMode.BICUBIC))
        transform.append(CenterCrop((200, 200)))
        transform = Compose(transform)

        return transform(img)
