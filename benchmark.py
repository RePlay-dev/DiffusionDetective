import os
from pathlib import Path
from typing import List, Dict, Type

import matplotlib.pyplot as plt
from fastai.vision.core import PILImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from detectors import FastAIDetector, ClipDetector, CorviDetector, Detector
from explainers import RISEExplainer, AblationCAMExplainer, SHAPExplainer
# The custom labeling function get_y needs to be imported in order to unpickle the FastAI model
# noinspection PyUnresolvedReferences
from utils import get_y


def compare_detectors_and_explainers(
        detectors: Dict[str, Type[Detector]],
        explainer_classes: Dict[str, Type],
        img_dirs: List[str] | None = None,
        output_dir: str = 'plots'
):
    if img_dirs is None:
        img_dirs = ['img/real', 'img/fake']

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize detectors
    detector_instances = {name: cls() for name, cls in detectors.items()}

    # Add explainers to each detector
    for detector in detector_instances.values():
        for name, explainer_class in explainer_classes.items():
            detector.add_explainer(name, explainer_class(detector.model))

    # Process all images in the specified directories
    for img_dir in img_dirs:
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img = PILImage.create(img_path)

                    print(f'Now processing image {img_path}')
                    for detector_name, detector in detector_instances.items():
                        process_image(img, detector, detector_name, img_path, output_dir)


def process_image(img: PILImage, detector: Detector, detector_name: str, img_path: str, output_dir: str):
    prediction, probability, _ = detector.get_prediction(img)
    explanations = detector.explain(img, list(detector.explainers.keys()))
    processed_image = detector.get_processed_image(img)

    # Calculate the number of subplots
    num_plots = len(explanations) + 1

    # Create a figure with subplots in one row
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    # Ensure axs is always a 1D array
    if num_plots == 1:
        axs = [axs]

    # Plot original image
    axs[0].imshow(processed_image)
    axs[0].set_title(f'Predicted: {prediction} ({probability:.2%})')
    axs[0].axis('off')

    # Add a dummy colorbar to the first image to maintain consistent sizing
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.axis('off')

    # Plot explanations
    for idx, (explainer_name, saliency_map) in enumerate(explanations.items(), start=1):
        ax = axs[idx]
        ax.imshow(processed_image)
        im = ax.imshow(saliency_map, cmap='jet', alpha=0.5)
        ax.set_title(explainer_name)
        ax.axis('off')

        # Add colorbar with consistent sizing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Add detector name as a centered super title
    fig.suptitle(f'{detector_name}', fontsize=16, y=1.05)

    # Generate filename
    img_relative_path = os.path.relpath(img_path, 'img')
    # Remove the original file extension
    img_relative_path_without_ext = os.path.splitext(img_relative_path)[0]
    # Replace '/' with '_' and ' ' with '-'
    sanitized_path = img_relative_path_without_ext.replace('/', '_').replace(' ', '-')
    filename = f"{detector_name}_{sanitized_path}.png"
    output_path = os.path.join(output_dir, filename)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# Usage
if __name__ == '__main__':
    detectors = {
        'FastAI': lambda: FastAIDetector('models/export_21_convnextv2_tiny_epoch_9.pkl'),
        'CLIP': ClipDetector,
        'Corvi': CorviDetector,
        # 'DIRE': lambda: DireDetector('models/lsun_adm.pth')
    }

    explainers = {
        'RISE': RISEExplainer,
        'AblationCAM': AblationCAMExplainer,
        'SHAP': SHAPExplainer
    }

    compare_detectors_and_explainers(detectors, explainers, img_dirs=['img/fake/midjourney_cifake/'])