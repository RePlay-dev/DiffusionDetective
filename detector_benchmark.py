import os
import time
from pathlib import Path

import pandas as pd
from fastai.vision.core import PILImage

from detectors import DiffusionDetectiveDetector, ClipDetector, CorviDetector
# The custom labeling function get_y needs to be imported in order to unpickle the DiffusionDetective model
# noinspection PyUnresolvedReferences
from utils import get_y

detectors = {
    'DiffusionDetective': DiffusionDetectiveDetector('models/diffusion_detective.pkl'),
    'CLIP': ClipDetector(),
    'Corvi': CorviDetector()
}


def detect_all_in_folder(root_folder: str, detector_name: str = 'DiffusionDetective'):
    """
    Runs a detector on all images in `root_folder`
    """
    detector = detectors[detector_name]

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                try:
                    img = PILImage.create(img_path)
                    prediction, probability, *_ = detector.get_prediction(img)
                    print(f'File: {img_path}')
                    print(f'Prediction: {prediction}')
                    print(f'Probability: {probability:.2%}')
                    print('-' * 50)
                except Exception as e:
                    print(f'Error processing {img_path}: {str(e)}')
                    print('-' * 50)


def benchmark(image_folder: Path, csv_filename='test_results.csv'):
    results = []

    for name, detector in detectors.items():
        print(f"\nProcessing {name} detector...")
        start_time = time.time()
        report, auc = detector.get_metrics(image_folder)
        end_time = time.time()
        processing_time = end_time - start_time

        results.append({
            'Detector': name,
            'Precision Fake': report['fake']['precision'],
            'Recall Fake': report['fake']['recall'],
            'Precision Real': report['real']['precision'],
            'Recall Real': report['real']['recall'],
            'f1-score Fake': report['fake']['f1-score'],
            'f1-score Real': report['real']['f1-score'],
            'Accuracy': report['accuracy'],
            'AUC': auc,
            'Processing Time (s)': processing_time
        })

    # Create DataFrame from all results
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(csv_filename, index=False)
    print(f"\nTest results saved to {csv_filename}")

    print(f"\nResults:")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    image_folder = Path('data/test/')
    benchmark(image_folder, csv_filename='test_results.csv')

    # detect_all_in_folder('data/test/', detector_name='DiffusionDetective')
