import torch
import torch.nn as nn
import math
from einops import rearrange

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.to(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.image_width = int(math.sqrt(input_dim))
        self.image_height = int(math.sqrt(input_dim))

        self.__setup_encoder()
        self.__setup_decoder()
    
    def __setup_encoder(self):
        self.enc_fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)

        self.relu = nn.ReLU().to(self.device)

        # Now we have two layers for each vector in latent space (going from hidden_dim to latent_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)  # Mean vector
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)  # Log-variance vector

    def __setup_decoder(self):
        self.dec_fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.dec_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
    
    def encode(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)') # Flatten the input
        x = self.relu(self.enc_fc1(x))
        x = self.relu(self.enc_fc2(x))

        mean = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        # Here we don't return x, we return mean and log_var, this is different to AE
        return mean, log_var
    
    def decode(self, latent):
        x = self.relu(self.dec_fc1(latent))
        x = self.relu(self.dec_fc2(x))

        x_hat = torch.sigmoid(self.dec_fc3(x))

        x_hat = rearrange(x_hat, 'b (c h w) -> b c h w', c=1, h=self.image_width, w=self.image_height) # Reshape the output

        return x_hat
    
    def reparameterization(self, mean, var):
        """
        Variance is exponential of log_var
        """
        epsilon = torch.randn_like(var).to(self.device)
        mean = mean.to(self.device)
        var = var.to(self.device)

        z = mean + var * epsilon
        return z


    def forward(self, x):
        assert x.shape[-3:] == (1, 28, 28)

        x.to(self.device)

        # Encode - instead of latent vector we get mean and log_var (look at image!)
        mean, log_var = self.encode(x)

        # Here is the magic of VAE
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        
        # Decode
        x_reconstructed = self.decode(z)

        # Return x hat
        return x_reconstructed, mean, log_var

class MNIST_VAE(VariationalAutoencoder):
    def __init__(self):
        super(MNIST_VAE, self).__init__(input_dim=1*28*28, hidden_dim=400, latent_dim=200)


class InterpolationModel(nn.Module):
    def __init__(self, vae_model):
        super(InterpolationModel, self).__init__()
        self.vae = vae_model
    
    def forward(self, input_img1: torch.Tensor, input_img2: torch.Tensor, interpolation: float) -> torch.Tensor:
        # Check if the input images size
        assert input_img1.shape == (1, 28, 28), "Input image 1 must be of shape (1, 28, 28)"
        assert input_img2.shape == (1, 28, 28), "Input image 2 must be of shape (1, 28, 28)"

        # Add batch dimension
        input_img1 = rearrange(input_img1, 'c h w -> 1 c h w') # Add batch dimension
        input_img2 = rearrange(input_img2, 'c h w -> 1 c h w') # Add batch dimension

        # Encode the two images
        mean1, log_var1 = self.vae.encode(input_img1)
        mean2, log_var2 = self.vae.encode(input_img2)

        # Reparameterization trick for both images
        z1 = self.vae.reparameterization(mean1, torch.exp(0.5 * log_var1))
        z2 = self.vae.reparameterization(mean2, torch.exp(0.5 * log_var2))

        # Interpolate between the two latent vectors
        latent_vector = (1 - interpolation) * z1 + interpolation * z2 # Linear interpolation

        # Decode interpolation to image
        interpolated_image = self.vae.decode(latent_vector)

        return interpolated_image.squeeze(0) # Remove the batch dimension only (keep channel dimension 1)

def save_interpolation_model():
    """
    Save the model to a file.
    """
    # Load VAE
    vae = VariationalAutoencoder(input_dim=1*28*28, hidden_dim=400, latent_dim=200)
    vae.load_state_dict(torch.load("vae_model.pth", map_location='cpu'))
    vae.to('cpu')
    vae.eval()

    # Create InterpolationModel
    model = InterpolationModel(vae)
    model.to('cpu')
    model.eval()

    # Save interpolation model
    print("Saving interpolation model...")
    torch.save(model.state_dict(), "interpolation_model.pth")

def interpolation_model_example():
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # Example usage of the InterpolationModel saved above
    model = InterpolationModel('cpu')
    model.load_state_dict(torch.load("interpolation_model.pth", map_location='cpu'))
    model.eval()  # Set the model to evaluation mode

    # Read images from local file
    transform = transforms.ToTensor()
    
    image = Image.open('frontend/public/mnist/1/1_1.png').convert('L')
    input_img1 = transform(image).to('cpu')

    image = Image.open('frontend/public/mnist/2/2_1.png').convert('L')
    input_img2 = transform(image).to('cpu')

    # Run the model - output is image
    interpolation = 0.9999999  # Interpolation factor (0.0 to 1.0)
    interpolated_image = model(input_img1.unsqueeze(0), input_img2.unsqueeze(0), interpolation)

    # Remove batch and channel dimensions for plotting
    img1 = input_img1.squeeze().numpy()
    img2 = input_img2.squeeze().numpy()
    interp = interpolated_image.squeeze().detach().cpu().numpy()

    # Plot the images in a row: img1, interpolated, img2
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(1, 3, 2)
    plt.imshow(interp, cmap='gray')
    plt.axis('off')
    plt.title('Interpolated')

    plt.subplot(1, 3, 3)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Image 2')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_interpolation_model()
    # interpolation_model_example()
