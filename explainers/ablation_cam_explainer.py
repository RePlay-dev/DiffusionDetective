import numpy as np
import torch
from pytorch_grad_cam import AblationCAM, GradCAM, GradCAMPlusPlus, EigenGradCAM, RandomCAM, AblationLayerVit, \
    AblationLayer

from explainers.base import Explainer


class AblationCAMExplainer(Explainer):
    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        print("Generating AblationCAM Explanation")
        # GradCam requires gradients to be calculated
        for p in self.model.parameters():
            p.requires_grad = True

        img_tensor.requires_grad = True

        # Determine the model type and set the target layer accordingly
        if hasattr(self.model, 'bb') and hasattr(self.model.bb[0], 'visual'):  # CLIP model
            target_layer = self.model.bb[0].visual.transformer.resblocks[-1]
            reshape_transform = self.clip_reshape_transform
        elif hasattr(self.model, 'layer4'):  # ResNet model (Corvi)
            target_layer = self.model.layer4[-1]
            reshape_transform = None
        elif hasattr(self.model[0], 'model'):  # ConvNeXT model (FastAI)
            convnext_model = self.model[0].model
            last_stage = convnext_model.stages[-1]
            last_block = last_stage.blocks[-1]
            target_layer = last_block.conv_dw
            reshape_transform = None
        else:
            raise ValueError("Unsupported model architecture")

        # Create an AblationCAM object using the correct target layer
        ablation_layer = AblationLayerVit() if reshape_transform else AblationLayer()
        ablation_cam = AblationCAM(
            model=self.model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
            ablation_layer=ablation_layer
        )

        # Generate the CAM mask
        ablation_cam_mask = ablation_cam(input_tensor=img_tensor, eigen_smooth=True, aug_smooth=True)[0, :]

        # Disable gradients for all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        return ablation_cam_mask

    @staticmethod
    def clip_reshape_transform(tensor):
        # Check if the tensor is already in the correct shape
        if len(tensor.shape) == 4:
            return tensor

        # For the shape [257, 1, 1024], we need to remove the second dimension and reshape
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)

        # Calculate height and width
        sequence_length = tensor.shape[0]
        height = width = int((sequence_length - 1) ** 0.5)

        # Reshape the tensor, excluding the first token (CLS token)
        result = tensor[1:].reshape(1, height, width, -1)

        # Bring the channels to the first dimension, like in CNNs
        result = result.permute(0, 3, 1, 2)
        return result

    def get_all_gradcam_saliency_maps(self, img_tensor: torch.Tensor) -> list[tuple[str, np.ndarray]]:
        print("Generating Explanations from all GradCAM Variants")
        # Enable gradients for parameters for GradCAM processing
        for p in self.model.parameters():
            p.requires_grad = True

        # Enable gradient
        img_tensor.requires_grad = True

        # Extract the target layer for CAM methods
        convnext_model = self.model[0].model
        last_stage = convnext_model.stages[-1]
        last_block = last_stage.blocks[-1]
        target_layer = last_block.conv_dw

        # Dictionary to hold GradCAM variants and their computed masks
        cams = {
            "GradCAM": GradCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "EigenGradCAM": EigenGradCAM,
            "AblationCAM": AblationCAM,
            "RandomCAM": RandomCAM
        }

        # Using list comprehension to initialize CAM objects and compute masks along with names
        cam_results = [
            (cam_name, cam_class(model=self.model, target_layers=[target_layer])
                       (input_tensor=img_tensor, eigen_smooth=True, aug_smooth=True)[0, :])
            for cam_name, cam_class in cams.items()
        ]

        # Disable gradients for parameters after processing
        for p in self.model.parameters():
            p.requires_grad = False

        return cam_results
