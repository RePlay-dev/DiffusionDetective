from pathlib import Path

import torch
from fastai.data.transforms import Category, get_image_files
from fastai.learner import load_learner
from fastai.torch_core import TensorImage
from fastai.vision.core import PILImage
from sklearn.metrics import classification_report, roc_auc_score

from detectors.base import Detector
from utils import device


class FastAIDetector(Detector):
    def __init__(self, model_path: str):
        self.learn = load_learner(model_path, cpu=(not torch.cuda.is_available()))
        self.model = self.learn.model
        super().__init__(self.model)

    def get_prediction(self, img: PILImage) -> tuple[Category, float, int]:
        result, index, prob = self.learn.predict(img)
        # For some reason, calling predict moves the model to the CPU
        # We have to ensure the model stays on the correct device so any operations after this stay fast
        self.model.to(device)
        print(f'FastAI detected the image as {result}')
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

    def get_metrics(self, path: Path) -> tuple[dict, float]:
        # Create test dataloader
        test_files = get_image_files(path)
        print(f"Number of test files: {len(test_files)}")
        test_dl = self.learn.dls.test_dl(test_files, with_labels=True)
        print(f"Test dataloader length: {len(test_dl)}")

        # Get predictions
        test_preds, test_targets = self.learn.get_preds(dl=test_dl)

        # Convert predictions to class labels
        pred_labels = test_preds.argmax(dim=1).numpy()
        true_labels = test_targets.numpy()

        # Get class names
        class_names = self.learn.dls.vocab

        print('Generate classification report')
        # Generate classification report
        report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4, output_dict=True)

        # Calculate AUC
        auc = roc_auc_score(true_labels, test_preds[:, 1].numpy())

        print(report)
        print(f'AUC: {auc}')

        return report, auc
