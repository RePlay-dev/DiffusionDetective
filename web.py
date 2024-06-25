import io
import math

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from fastai.vision.core import PILImage

from detectors import FastAIDetector, DireDetector, ClipDetector, CorviDetector
from explainers import RISEExplainer, SHAPExplainer, AblationCAMExplainer
# The custom labeling function get_y needs to be imported in order to unpickle the FastAI model
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
    if approach == 'FastAI':
        detector = FastAIDetector('models/export_21_convnextv2_tiny_epoch_9.pkl')
    elif approach == 'DIRE':
        detector = DireDetector('models/lsun_adm.pth')
    elif approach == 'Clip':
        detector = ClipDetector()
    elif approach == 'Corvi':
        detector = CorviDetector()
    else:
        return "Invalid approach selected", 0, None

    img = PILImage.create(image)

    detector.add_explainer('RISE', RISEExplainer(detector.model))
    detector.add_explainer('SHAP', SHAPExplainer(detector.model))
    detector.add_explainer('AblationCAM', AblationCAMExplainer(detector.model))

    prediction, probability, _ = detector.get_prediction(image)
    explanations = detector.explain(image, explainers)

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
            gr.Radio(choices=["FastAI", "DIRE", "Clip", "Corvi"], value="FastAI", label="Detection Approach",
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
        title="Image Detection and Explanation",
        description="Upload an image to detect if it's 'real' or 'fake' using either FastAI or DIRE approach.",
        examples=[
            ["img/dalle2_r00444b95t.png", "FastAI", "RISE",
             "Image generated with DALL·E 2. From the Synthbuster dataset."],
            ["img/dalle3_r08996953t.png", "FastAI", ["RISE", "AblationCAM"],
             "Image generated with DALL·E 3. From the Synthbuster dataset."],
            ["img/firefly_r02b3a1ddt.png", "FastAI", "AblationCAM",
             "Image generated with Adobe Firefly. From the Synthbuster dataset."],
            ["img/glide_r0f979863t.png", "FastAI", "RISE", "Image generated with GLIDE. From the Synthbuster dataset."],
            ["img/midjourney-v5_r0773471dt.png", "FastAI", "RISE",
             "Image generated with Midjourney. From the Synthbuster dataset."],
            ["img/stable-diffusion-1-3_r00816405t.png", "FastAI", "RISE",
             "Image generated with Stable Diffusion 1.3. From the Synthbuster dataset."],
            ["img/stable-diffusion-1-4_r09488b39t.png", "FastAI", "RISE",
             "Image generated with Stable Diffusion 1.4. From the Synthbuster dataset."],
            ["img/stable-diffusion-2_r018ba134t.png", "FastAI", "RISE",
             "Image generated with Stable Diffusion 2. From the Synthbuster dataset."],
            ["img/stable-diffusion-xl_r01b3f004t.png", "FastAI", "RISE",
             "Image generated with Stable Diffusion XL. From the Synthbuster dataset."],
            ["img/real_flickr_1001465944.jpg", "FastAI", "RISE", "Real image from the Flickr30k dataset."],
            ["img/real_raise_r0d0ff43at-0.png", "FastAI", "RISE", "Real image from the RAISE-1k dataset."],
            ["img/dire-val_lsun-bedroom_adm_0.png", "DIRE", "RISE",
             "Image generated with ?, trained on the LSUN Bedroom dataset."]
        ],
        allow_flagging="never",
        cache_examples=False
    )

    # Run the interface
    iface.launch()
