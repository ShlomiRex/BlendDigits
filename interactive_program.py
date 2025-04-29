import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.datasets import MNIST
import tkinter as tk
from PIL import Image, ImageTk
from torch import Tensor
import tkinter.messagebox as messagebox
from model import InterpolationModel, MNIST_VAE, MNISTInterpolationModel
from typing import Tuple

batch_size = 16

def load_model() -> InterpolationModel:
    device = 'cpu'
    print(f"Using device: {device}")

    # Load the interpolation model
    vae_model_path = "vae_model.pth"
    interpolation_model = MNISTInterpolationModel(vae_model_path, device).to(device).eval()
    
    return interpolation_model

def load_mnist() -> Tuple[DataLoader, MNIST]:
    # Load mnist data
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist = MNIST(root="./data", train=False, transform=transform, download=True)
    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    print("MNIST dataset loaded successfully.")

    return mnist_loader, mnist
# Create TkInter App
class App:
    def __init__(self, root):
        self.interpolation_model = load_model()
        self.mnist_loader, self.mnist = load_mnist()

        self.root = root
        self.root.title("Interactive Interpolation")
        self.root.geometry("400x200")

        # Store the tensors of the images
        self.img1_tensor = self.sample_image()
        self.img2_tensor = self.sample_image()
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

    def sample_image(self) -> Tensor:
        # Sample a single image from the dataset
        num_of_images = len(self.mnist)
        i = torch.randint(low=0, high=(num_of_images-1), size=(1,)).item()
        return self.mnist_loader.dataset[i][0]

    def new_images(self):
        self.img1_tensor = self.sample_image()
        self.img2_tensor = self.sample_image()

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
        # Run the model - output is image
        interpolated_image = self.interpolation_model(self.img1_tensor, self.img2_tensor, interpolation)

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

