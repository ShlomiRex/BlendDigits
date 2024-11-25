import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.datasets import MNIST
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from torch import Tensor
import tkinter.messagebox as messagebox

# Load the VAE model
input_dim = 1 * 28 * 28
hidden_dim = 400
latent_dim = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.image_width = int(math.sqrt(input_dim))
        self.image_height = int(math.sqrt(input_dim))

        self.__setup_encoder()
        self.__setup_decoder()
    
    def __setup_encoder(self):
        self.enc_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.relu = nn.ReLU()

        # Now we have two layers for each vector in latent space (going from hidden_dim to latent_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)  # Mean vector
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)  # Log-variance vector

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
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z


    def forward(self, x):
        assert x.shape[-3:] == (1, 28, 28)

        # Encode - instead of latent vector we get mean and log_var (look at image!)
        mean, log_var = self.encode(x)

        # Here is the magic of VAE
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        
        # Decode
        x_reconstructed = self.decode(z)

        # Return x hat
        return x_reconstructed, mean, log_var

# Load the model
vae = VariationalAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()

# Check the model
print(vae)

# Load mnist data
transform = transforms.Compose([
    transforms.ToTensor(),
])
mnist = MNIST(root="./data", train=False, transform=transform, download=True)
mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

def sample_image() -> Tensor:
    # Sample a single image from the dataset
    num_of_images = len(mnist)
    i = torch.randint(low=0, high=(num_of_images-1), size=(1,)).item()
    return mnist_loader.dataset[i][0]

# Create TkInter App
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Interpolation")
        self.root.geometry("400x200")

        # Store the tensors of the images
        self.img1_tensor = sample_image()
        self.img2_tensor = sample_image()
        self.interpolated_tensor = None # Placeholder for the interpolated tensor

        # Display the images
        self.update_images()

        # Add button below the images
        button = tk.Button(self.root, text="New images", command=self.new_images)
        button.grid(row=1, column=0, columnspan=3, pady=5)

        # Add slider between 0 and 1
        slider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, command=self.slider_updated)
        slider.set(0.5)
        slider.grid(row=2, column=0, columnspan=3, pady=5)

    def new_images(self):
        self.img1_tensor = sample_image()
        self.img2_tensor = sample_image()

        self.update_images()
        self.update_interpolation(self.slider_value)
    
    def update_images(self):
        # Display the first image
        img1 = Image.fromarray((self.img1_tensor * 255).squeeze().numpy())
        img1 = img1.resize((112, 112), Image.Resampling.NEAREST)
        img1_tk = ImageTk.PhotoImage(img1)
        label1 = tk.Label(self.root, image=img1_tk)
        label1.image = img1_tk
        label1.grid(row=0, column=0, padx=5)

        # Display the second image
        img2 = Image.fromarray((self.img2_tensor * 255).squeeze().numpy())
        img2 = img2.resize((112, 112), Image.Resampling.NEAREST)
        img2_tk = ImageTk.PhotoImage(img2)
        label2 = tk.Label(self.root, image=img2_tk)
        label2.image = img2_tk
        label2.grid(row=0, column=2, padx=5)


    def slider_updated(self, value):
        value = float(value)
        self.slider_value = value
        self.update_interpolation(value)
    
    def update_interpolation(self, interpolation: float):
        # Add batch dimension of 1
        img1_tensor = rearrange(self.img1_tensor, "c h w -> 1 c h w")
        img2_tensor = rearrange(self.img2_tensor, "c h w -> 1 c h w")

        # The model is loaded with CUDA probably, and image tensors are on CPU, we have to match either of them
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # Get first latent vector
        mean, log_var = vae.encode(img1_tensor)
        z1 = vae.reparameterization(mean, torch.exp(0.5 * log_var))

        # Get second latent vector
        mean, log_var = vae.encode(img2_tensor)
        z2 = vae.reparameterization(mean, torch.exp(0.5 * log_var))

        # Interpolate between the two latent vectors
        latent_vector = (1 - interpolation) * z1 + interpolation * z2 # Linear interpolation

        # Decode interpolation to image
        interpolated_image = vae.decode(latent_vector)

        # Update the image
        interpolated_image = interpolated_image.squeeze(0) # Remove the batch dimension only (keep channel dimension 1)
        interpolated_image = (interpolated_image * 255) # Fit to RGB scale
        self.interpolated_tensor = interpolated_image

        # Convert to PIL image
        interpolated_image = interpolated_image.cpu().detach().numpy()
        interpolated_image = interpolated_image.squeeze() # Also remove the channel dimension for PIL
        interpolated_image = Image.fromarray(interpolated_image)
        interpolated_image = interpolated_image.resize((112, 112), Image.Resampling.NEAREST)

        # Update the image in TkInter
        interpolated_image_tk = ImageTk.PhotoImage(interpolated_image)
        interpolated_label = tk.Label(self.root, image=interpolated_image_tk)
        interpolated_label.image = interpolated_image_tk
        interpolated_label.grid(row=0, column=1, padx=5)





if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)

    def check():
        root.after(50, check)

    root.after(50, check)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(e)

