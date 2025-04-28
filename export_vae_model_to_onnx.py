import torch

from interactive_interpolation import VariationalAutoencoder

# Load the VAE model
input_dim = 1 * 28 * 28
hidden_dim = 400
latent_dim = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# Load the model
vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
vae.eval()

# Create dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# Export
torch.onnx.export(
    vae,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print("Exported to model.onnx!")
