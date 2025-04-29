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

class InterpolationModel(VariationalAutoencoder):
    def __init__(self):
        # Default parameters for the VAE model
        input_dim = 1 * 28 * 28
        hidden_dim = 400
        latent_dim = 200
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
        vae.load_state_dict(torch.load("vae_model.pth", map_location=device))
        vae.eval() # Set the model to evaluation mode, we don't want to train it again
        vae.to(device)
        super(InterpolationModel, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, device=device)

        self.vae = vae

    
    def forward(self, input_img1: torch.Tensor, input_img2: torch.Tensor, interpolation: float) -> torch.Tensor:
        # Encode the two images
        mean1, log_var1 = self.encode(input_img1)
        mean2, log_var2 = self.encode(input_img2)

        # Reparameterization trick for both images
        z1 = self.reparameterization(mean1, torch.exp(0.5 * log_var1))
        z2 = self.reparameterization(mean2, torch.exp(0.5 * log_var2))

        # Interpolate between the two latent vectors
        latent_vector = (1 - interpolation) * z1 + interpolation * z2 # Linear interpolation

        # Decode interpolation to image
        interpolated_image = self.vae.decode(latent_vector)

        return interpolated_image.squeeze(0) # Remove the batch dimension only (keep channel dimension 1)

if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # Example usage
    model = InterpolationModel()

    # Read images from local file
    transform = transforms.ToTensor()
    
    image = Image.open('frontend/public/mnist/1/1_1.png').convert('L')
    input_img1 = transform(image).to('cpu')

    image = Image.open('frontend/public/mnist/2/2_1.png').convert('L')
    input_img2 = transform(image).to('cpu')

    # Run the model - output is image
    interpolation = 0.5  # Interpolation factor (0.0 to 1.0)
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