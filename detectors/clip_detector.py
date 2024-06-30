from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from fastai.vision.core import PILImage
from sklearn.metrics import roc_auc_score, classification_report
from torch import Tensor, load
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from tqdm import tqdm

from detectors.base import Detector
from detectors.corvi_detector import CorviTestDataset
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

        print(f'Clip detected the image as {prediction}')
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

    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def get_metrics(self, path: Path, batch_size=32) -> tuple[dict, float]:
        # Create test dataset and dataloader
        test_transform = self.get_test_transform()
        test_dataset = CorviTestDataset(root_dir=path, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"Number of test files: {len(test_dataset)}")
        print(f"Number of batches: {len(test_loader)}")

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing batches"):
                images, targets = batch
                images = images.to(device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Convert predictions to class labels
        pred_labels = (all_preds <= 0.5).astype(int)

        # Generate classification report
        class_names = ['fake', 'real']

        # Calculate AUC
        auc = roc_auc_score(all_targets, 1 - all_preds)

        report = classification_report(all_targets, pred_labels, target_names=class_names, digits=4, output_dict=True)

        print(report)
        print(f'AUC: {auc}')

        return report, auc
