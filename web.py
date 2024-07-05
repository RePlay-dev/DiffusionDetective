import io
import math

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from fastai.vision.core import PILImage

from detectors import DiffusionDetectiveDetector, ClipDetector, CorviDetector
from explainers import RISEExplainer, SHAPExplainer, AblationCAMExplainer
# The custom labeling function get_y needs to be imported in order to unpickle the DiffusionDetective model
# noinspection PyUnresolvedReferences
from utils import get_y


def convert_explanations_to_image(input_img, explanations):
    # Calculate the number of rows needed for the subplots.
    num_explanations = len(explanations)
    num_rows = math.ceil(num_explanations / 2)

    # Create a figure with subplots. figsize can be adjusted based on the number of explanations.
    plt.figure(figsize=(10, 5 * num_rows))

    # Iterate over each explanation in the dictionary.
    for index, (algorithm, saliency_map) in enumerate(explanations.items(), 1):
        # Create a subplot for each explanation.
        plt.subplot(num_rows, 2, index)
        plt.axis('off')
        plt.title(f'{algorithm}')

        # Display the input image.
        plt.imshow(input_img)
        # Overlay the explanation image with a colormap and set the transparency (alpha).
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    # Adjust layout to make sure there is no overlap and everything is nicely spaced.
    plt.tight_layout()

    # Convert the plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close()

    return image


def detect_image(image, approach, explainers, info):
    if approach == 'DiffusionDetective':
        detector = DiffusionDetectiveDetector('models/diffusion_detective.pkl')
    elif approach == 'CLIP':
        detector = ClipDetector()
    elif approach == 'Corvi':
        detector = CorviDetector()
    else:
        return "Invalid approach selected", 0, None

    img = PILImage.create(image)

    detector.add_explainer('RISE', RISEExplainer(detector.model))
    detector.add_explainer('SHAP', SHAPExplainer(detector.model))
    detector.add_explainer('AblationCAM', AblationCAMExplainer(detector.model))

    prediction, probability, _ = detector.get_prediction(img)
    explanations = detector.explain(img, explainers)

    processed_image = detector.get_processed_image(img)

    # Convert saliency map to an image
    saliency_image = convert_explanations_to_image(processed_image, explanations)

    return prediction, f"{probability:.2%}", saliency_image


if __name__ == '__main__':
    # Define the Gradio interface
    iface = gr.Interface(
        fn=detect_image,
        inputs=[
            gr.Image(label="Input Image"),
            gr.Radio(choices=["DiffusionDetective", "CLIP", "Corvi"], value="DiffusionDetective",
                     label="Detection Approach",
                     info="Choose an approach"),
            gr.CheckboxGroup(choices=["RISE", "AblationCAM", "SHAP"], value="RISE", label="Explainer",
                             info="Choose one or more explainers"),
            gr.Textbox(label="Info", visible=False)
        ],
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Probability"),
            gr.Image(label="Saliency Map")
        ],
        title="DiffusionDetective",
        description="Upload an image to detect if it's 'real' or 'fake' using the DiffusionDetective, CLIP, or Corvi approach. This detection can be explained by the RISE, AblationCAM and SHAP explainers.",
        examples=[
            ["img/fake/midjourney_cifake/0100 (4).jpg", "DiffusionDetective", "RISE",
             "Image generated with Midjourney v6. From the Midjourney CIFAKE-Inspired dataset."],
            ["img/fake/synthbuster/sd1-3-r0a2ff882t.png", "CLIP", ["RISE", "SHAP"],
             "Image generated with Stable Diffusion 1.3. From the Synthbuster dataset."],
            ["img/fake/stable-imagenet1k/010_3940.jpg", "Corvi", ["AblationCAM", "SHAP"],
             "Image generated with Stable Diffusion 1.4. From the Stable ImageNet-1K dataset."],
            ["img/real/raise/r0a966704t.png", "DiffusionDetective", ["RISE", "AblationCAM"],
             "Real image from the RAISE-1k dataset."],
            ["img/real/flickr/371897.jpg", "CLIP", "SHAP",
             "Real image from the Flickr30k dataset."],
            ["img/real/imagenet1k/00012_025771816061583.jpg", "Corvi", ["RISE", "SHAP"],
             "Real image from the ImageNet-1k-valid dataset."],
        ],
        allow_flagging="never",
        cache_examples=False
    )

    # Run the interface
    iface.launch()
