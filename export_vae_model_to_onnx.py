import torch

from model import MNISTInterpolationModel

# Load the model
model_path = "interpolation_model.pth"
model = MNISTInterpolationModel(model_path).eval()

# Print state dict
print(model.state_dict)

# Create dummy input
dummy_img1 = torch.randn(1, 28, 28)
dummy_img2 = torch.randn(1, 28, 28)
dummy_interpolation = 0.5

# Export
torch.onnx.export(
    model,
    (dummy_img1, dummy_img2, dummy_interpolation),
    "model.onnx",
    input_names=['input_img1', 'input_img2', 'interpolation'],
    output_names=['output'],
    opset_version=11
)

print("Exported to model.onnx!")
