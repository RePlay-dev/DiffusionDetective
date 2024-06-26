import math

from fastai.vision.core import PILImage
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from detectors import FastAIDetector
from explainers import RISEExplainer, AblationCAMExplainer, SHAPExplainer
# The custom labeling function get_y needs to be imported in order to unpickle the FastAI model
# noinspection PyUnresolvedReferences
from utils import get_y


def gradcam_test(detector, input_img):
    # Extracting saliency maps along with their names
    processed_tensor = detector.get_processed_tensor(input_img)
    cam_results = detector.explainers["AblationCAM"].get_all_gradcam_saliency_maps(processed_tensor)

    # Preprocessing the input image for display
    processed_image = detector.get_processed_image(input_img)

    # Getting model predictions
    prediction, probability, index = detector.get_prediction(input_img)

    # Setting up the plot
    plt.figure(figsize=(10, 10))

    # Displaying the processed image with prediction info
    plt.subplot(3, 2, 1)
    plt.axis('off')
    plt.title(f'Predicted: {prediction} ({probability:.2%})')
    plt.imshow(processed_image)

    # Displaying all saliency maps using a loop
    for i, (title, saliency_map) in enumerate(cam_results, start=2):
        plt.subplot(3, 2, i)
        plt.axis('off')
        plt.title(title)
        plt.imshow(processed_image)  # Show the original image
        plt.imshow(saliency_map, cmap='jet', alpha=0.5)  # Overlay the saliency map
        plt.colorbar(fraction=0.046, pad=0.04)  # Show color bar for interpretation

    plt.show()


def matplotlib_show(input_img, explanations, prediction, probability, fname=''):
    # Calculate the number of rows needed for the subplots
    num_explanations = len(explanations)
    num_rows = math.ceil((num_explanations + 1) / 2)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot the input image
    im = axs[0].imshow(input_img)
    axs[0].set_title(f'Predicted: {prediction} ({probability:.2%})')
    axs[0].axis('off')

    # Add a dummy colorbar to the first image to maintain consistent sizing
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.axis('off')

    # Iterate over each explanation in the dictionary
    for index, (algorithm, saliency_map) in enumerate(explanations.items(), 1):
        ax = axs[index]
        ax.imshow(input_img)
        im = ax.imshow(saliency_map, cmap='jet', alpha=0.5)
        ax.set_title(f'{algorithm}')
        ax.axis('off')

        # Add colorbar with consistent sizing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Remove any unused subplots
    for i in range(num_explanations + 1, len(axs)):
        fig.delaxes(axs[i])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if a filename is provided
    if fname:
        plt.savefig(f'plots/{fname}', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    detector = FastAIDetector('models/export_21_convnextv2_tiny_epoch_9.pkl')
    # detector = ClipDetector()
    # detector = CorviDetector()

    img = PILImage.create('img/fake/midjourney_cifake/0100 (4).jpg')
    # img = PILImage.create('img/real/flickr/371897.jpg')
    # img = PILImage.create('img/fake/synthbuster/midjourney-r0a0852c6t.png')

    # detector = DireDetector('models/lsun_adm.pth')
    # img = PILImage.create('img/dire-val_lsun-bedroom_adm_0.png')

    explainers = ['RISE', 'AblationCAM', 'SHAP']
    detector.add_explainer('RISE', RISEExplainer(detector.model))
    detector.add_explainer('SHAP', SHAPExplainer(detector.model))
    detector.add_explainer('AblationCAM', AblationCAMExplainer(detector.model))

    prediction, probability, _ = detector.get_prediction(img)
    explanations = detector.explain(img, explainers)

    processed_image = detector.get_processed_image(img)

    matplotlib_show(processed_image, explanations, prediction, probability, fname='fastai_export_21-1')

    # gradcam_test(detector, img)
