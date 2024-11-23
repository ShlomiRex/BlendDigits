import torch.nn as nn
from einops import rearrange

"""
Autoencoder model. Not VAE. Used to compare to VAE.
"""

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.__setup_encoder()
        self.__setup_decoder()
    
    def __setup_encoder(self):
        self.enc_conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=2, padding=1) # Output: 512 x 14 x 14
        self.enc_relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1) # Output: 256 x 7 x 7
        self.enc_relu2 = nn.ReLU()
        self.enc_conv3 = nn.Conv2d(256, 128, kernel_size=7) # Output: 128 x 1 x 1
        self.enc_linear = nn.Linear(128, self.latent_dim) # Output: 1 x latent_dim

    def __setup_decoder(self):
        self.dec_linear = nn.Linear(self.latent_dim, 128) # Output: 1 x 128
        self.dec_conv1 = nn.ConvTranspose2d(128, 256, kernel_size=7) # Output: 512 x 7 x 7
        self.dec_relu1 = nn.ReLU()
        self.dec_conv2 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 512 x 14 x 14
        self.dec_relu2 = nn.ReLU()
        self.dec_conv3 = nn.ConvTranspose2d(512, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # Output: 1 x 28 x 28
        self.dec_tanh = nn.Tanh()
    
    def encode(self, x):
        assert x.shape[-3:] == (1, 28, 28)
        x = self.enc_conv1(x)
        x = self.enc_relu1(x)
        x = self.enc_conv2(x)
        x = self.enc_relu2(x)
        x = self.enc_conv3(x)
        x = rearrange(x, 'b c h w -> b (c h w)') # Remove h, w dimensions which are 1
        x = self.enc_linear(x)
        return x
    
    def decode(self, latent):
        assert latent.shape[-1] == self.latent_dim
        x = self.dec_linear(latent)
        x = rearrange(x, 'b c -> b c 1 1') # Add h, w dimensions which are 1, prepare to add spatial information
        x = self.dec_conv1(x)
        x = self.dec_relu1(x)
        x = self.dec_conv2(x)
        x = self.dec_relu2(x)
        x = self.dec_conv3(x)
        x = self.dec_tanh(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        x_reconstructed = self.decode(latent)
        return x_reconstructed