import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, datasets
from scipy.stats import norm
from sklearn.manifold import TSNE
import streamlit as st

# Download Dataset

train_transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

######## Just checking out what the dataset looks like

dataiter = iter(trainloader)
images, labels = next(dataiter)
images.shape
# plt.imshow(images[32].squeeze().numpy(), cmap='bone')

########Definition of the architecture of our encoder and decoder model with all the assisting functions

class Net(nn.Module):
    def __init__(self, num_latent):
        super().__init__()
        
        #So here we will first define layers for encoder network
        self.encoder = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1),
                                     nn.MaxPool2d(2, 2),
                                     nn.BatchNorm2d(3),
                                     nn.Conv2d(3, 16, 3, padding=1),
                                     nn.MaxPool2d(2, 2),
                                     nn.BatchNorm2d(16),
                                     nn.Conv2d(16, 16, 3, padding=1))
        
        #These two layers are for getting logvar and mean
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, num_latent)
        self.var = nn.Linear(128, num_latent)
        
        #######The decoder part
        #This is the first layer for the decoder part
        self.expand = nn.Linear(num_latent, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 784)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 16, 3, padding=1),
                                     nn.BatchNorm2d(16),
                                     nn.ConvTranspose2d(16, 3, 8),
                                     nn.BatchNorm2d(3),
                                     nn.ConvTranspose2d(3, 1, 15))
        
    def enc_func(self, x):
        #here we will be returning the logvar(log variance) and mean of our network
        x = self.encoder(x)
        x = x.view([-1, 784])
        x = F.dropout2d(self.fc1(x), 0.5)
        x = self.fc2(x)
        
        mean = self.mean(x)
        logvar = self.var(x)
        return mean, logvar
    
    def dec_func(self, z):
        #here z is the latent variable state
        z = self.expand(z)
        z = F.dropout2d(self.fc3(z), 0.5)
        z = self.fc4(z)
        z = z.view([-1, 16, 7, 7])        
        out = self.decoder(z)
        out = F.sigmoid(out)
        return out
    
    def get_hidden(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)   #So as to get std
            noise = torch.randn_like(mean)   #So as to get the noise of standard distribution
            return noise.mul(std).add_(mean)
        else:
            return mean
    
    def forward(self, x):
        mean, logvar = self.enc_func(x)
        z = self.get_hidden(mean, logvar)
        out = self.dec_func(z)
        return out, mean, logvar
    

    #######This is the custom loss function defined for VAE
### You can even refere to: https://github.com/pytorch/examples/pull/226 

def VAE_loss(out, target, mean, logvar):
    category1 = nn.BCELoss()
    bce_loss = category1(out, target)
    
    #We will scale the following losses with this factor
    scaling_factor = out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]
    
    ####Now we are gonna define the KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    kl_loss /= scaling_factor
    
    return bce_loss + kl_loss

######The function which we will call for training our model

def train(trainloader, iters, model, device, optimizer, print_every):
    counter = 0
    for i in range(iters):
        model.train()
        model.to(device)
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            out, mean, logvar = model(images)
            loss = VAE_loss(out, images, mean, logvar)
            loss.backward()
            optimizer.step()
            
        if(counter % print_every == 0):
            model.eval()
            n = 10  # figure with 20x20 digits
            digit_size = 28
            figure = np.zeros((digit_size * n, digit_size * n))

            # Construct grid of latent variable values
            grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

            counter = 0
            # decode for each square in the grid
            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    digit = out[counter].squeeze().cpu().detach().numpy()
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit
                    counter += 1

            plt.figure(figsize=(10, 10))
            plt.imshow(figure, cmap='bone')
            st.pyplot()

        counter += 1
                    
    ######Setting all the hyperparameters
##You can change them if you want

def load_model(num_latent,iters):
    model = Net(num_latent)
    # Load the saved weights
    model.load_state_dict(torch.load('vae_model_weights'+str(iters)+'.pth'))
    print("Model weights loaded.")
    return model


def show_original_images(model):
    from scipy.stats import norm

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display a 2D manifold of the digits
    n = 10  # figure with 20x20 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = images[counter].squeeze()
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='bone')
    plt.title('original image')
    plt.show() 
    st.pyplot(plt.gcf ())

    model.to('cpu')
    model.eval()
    out, _, _ = model(images)

    # Display a 2D manifold of the digits
    n = 10  # figure with 20x20 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = out[counter].squeeze().detach().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='bone')
    plt.title('generated image')
    st.pyplot(plt.gcf ())    


def combine_results_show(num_latent):
    iters = 1
    model = load_model(num_latent, iters)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display a 2D manifold of the digits
    n = 10  # figure with 20x20 digits
    digit_size = 28
    figure1 = np.zeros((digit_size * n, digit_size * n))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = images[counter].squeeze()
            figure1[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

    model.to('cpu')
    model.eval()
    out, _, _ = model(images)

    # Display a 2D manifold of the digits
    figure2 = np.zeros((digit_size * n, digit_size * n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = out[counter].squeeze().detach().numpy()
            figure2[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

######
    iters = 26
    model = load_model(num_latent, iters)

    model.to('cpu')
    model.eval()
    out, _, _ = model(images)

    # Display a 2D manifold of the digits
    figure3 = np.zeros((digit_size * n, digit_size * n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = out[counter].squeeze().detach().numpy()
            figure3[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit
            counter += 1

    iters = 50


    # Display a 2D manifold of the digits
    n = 10  # figure with 20x20 digits
    digit_size = 28

    model.to('cpu')
    model.eval()
    out, _, _ = model(images)

    # Display a 2D manifold of the digits
    figure4 = np.zeros((digit_size * n, digit_size * n))

    counter = 0
    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = out[counter].squeeze().detach().numpy()
            figure4[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit
            counter += 1



    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(16, 8))

    # Plot the first image
    axs[0].imshow(figure1, cmap='bone')
    axs[0].set_title('Original Images')

    # Plot the second image
    axs[1].imshow(figure2, cmap='bone')
    axs[1].set_title('1 iterations')

    # Plot the second image
    axs[2].imshow(figure3, cmap='bone')
    axs[2].set_title('26 iterations')

    # Plot the second image
    axs[3].imshow(figure4, cmap='bone')
    axs[3].set_title('50 iterations')


    # Adjust layout
    st.pyplot(fig)
    st.markdown("We can see drastic improvement from iteration 1 to 26 but not much from 26 to 50.")


# Function to visualize the latent space
def visualize_latent_space(model):
    st.text("Visualizing the latent space...wait it might take some time...")

    trainloader_new = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    dataiter = iter(trainloader_new)
    images, labels = next(dataiter)

    mean, logvar  = model.enc_func(images)
    z = model.get_hidden(mean, logvar)
    
    # Since my latent state consists of 8 dimensions, I must first reduce it to 2 dimensions
    # so as to be able to visualize them
    z_embedded = TSNE(n_components=2).fit_transform(z.detach().numpy())

    # Visualization of the latent space
    plt.figure(figsize=(8, 8))
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels, cmap='viridis')
    plt.title('Latent Space Visualization')
    st.pyplot(plt.gcf ())
    st.markdown("t-SNE plot help us to visualize 8 dimensional latent space in 2D. If we see the different clear clusters we can conclude that VAE has learned well the latent representations.")


def on_generate_button_click(model):
    st.write("Button was pressed!")
    rand_z = torch.randn([1, 8])
    out = model.dec_func(rand_z)
    plt.imshow(out.squeeze().detach().numpy(), cmap='bone')
    plt.title('new number')
    st.pyplot(plt.gcf ()) 
    st.markdown("Unknown digit.")

# Streamlit app

def main():
    st.title("Demo on Variational Autoencoders")
    st.subheader("Saumya Karan")
    # Training parameters
    iters_options = [1, 26, 50]
    
    print_every= 5

    st.markdown("Choose number of iterations of training. The number of latent variables is 8.")

    iters = st.radio("Number of iterations", iters_options)
    num_latent = 8

    st.markdown("Press this button to upload model trained for choosen no of iteration and see grnerated results. This also shows the latent space of VAE using t-SNE function.")
    loadmodelpressed=st.button("Load Pre-trained Model")
    st.markdown("Press this button to start training and then see results.")
    trainmodelpressed=st.button("Train_Instead")
    st.markdown("Press this button to see Magic!! Create your own digit by altering the latent space variables and decoding it.")
    generate_button_pressed = st.button("Generate Brand New Digit!")
    st.markdown("Press this button to see combined results of VAE output for different number of iterations.")
    combine_results_pressed = st.button("Show Combine Results")
    # Check if model weights are available
    if loadmodelpressed:
        st.text("Loading pre-trained model...")
        model = load_model(num_latent,iters)
        st.text("Model loaded successfully.")
        show_original_images(model)
        if generate_button_pressed:
            on_generate_button_click(model)
        visualize_latent_space(model)
    if trainmodelpressed:
        # Train the model
        st.text("Training the VAE model...")
        model = Net(num_latent)
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(trainloader, iters, model, device, optimizer, print_every)
        # Save the trained model weights
        st.text("Saving the trained model weights...")
        torch.save(model.state_dict(), 'vae_model_weights'+str(iters)+'.pth')
        st.text("Model weights saved.")
        visualize_latent_space(model)
    if generate_button_pressed:
        model = load_model(num_latent,iters)
        on_generate_button_click(model)
        
    if combine_results_pressed:
        
        combine_results_show(num_latent)
        
    
    generate_button_pressed = 0
    trainmodelpressed = 0
    loadmodelpressed = 0
    combine_results_pressed = 0
    # Display images from the dataset
    

    # Add a button to the app
    

# Check if the button is pressed
    

    # reconstructed_output(model)
    # Display latent space visualization
    

if __name__ == "__main__":
    main()






