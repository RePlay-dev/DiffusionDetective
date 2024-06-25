from fastai.data.transforms import Category
from fastai.learner import load_learner
from fastai.torch_core import TensorImage
from fastai.vision.core import PILImage

from detectors.base import Detector
from utils import device


class FastAIDetector(Detector):
    def __init__(self, model_path: str):
        self.learn = load_learner(model_path)
        self.model = self.learn.model
        super().__init__(self.model)

    def get_prediction(self, img: PILImage) -> tuple[Category, float, int]:
        result, index, prob = self.learn.predict(img)
        # For some reason, calling predict moves the model to the CPU
        # We have to ensure the model stays on the correct device so any operations after this stay fast
        self.model.to(device)
        return result, prob[index.item()].item(), index.item()

    def get_processed_tensor(self, img: PILImage) -> TensorImage:
        data_loader = self.learn.dls.test_dl([img], bs=1)  # Create a test DataLoader
        return data_loader.one_batch()[0].to(device)  # Retrieve the processed image as a tensor

    def get_processed_image(self, img: PILImage) -> TensorImage:
        data_loader = self.learn.dls.test_dl([img], bs=1)  # Create a test DataLoader
        # Retrieve the processed image in a format that can be displayed
        img_processed = data_loader.show_batch(show=False)[0]
        img_processed = img_processed.squeeze(0)
        # Permute the tensor dimensions from (3, 224, 224) to (224, 224, 3)
        return img_processed.permute(1, 2, 0)
