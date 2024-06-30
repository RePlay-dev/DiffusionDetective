from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from fastai.vision.core import PILImage
from sklearn.metrics import classification_report, roc_auc_score
from torch import Tensor, load
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop
from tqdm import tqdm

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

    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(200, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


class CorviTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
