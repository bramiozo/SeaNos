'''
Code for VAE to transform one mel spectrogram (speech) into another (song).


'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw


class SpectrogramAligner:
    def __init__(self, metric='euclidean'):
        self.metric = metric

    def align(self, source_spec, target_spec):
        """
        Align source spectrogram to target spectrogram using DTW.

        :param source_spec: numpy array of shape (time_steps, n_mels)
        :param target_spec: numpy array of shape (time_steps, n_mels)
        :return: aligned_source, aligned_target (numpy arrays)
        """
        # Compute DTW
        distance, path = fastdtw(source_spec, target_spec, dist=self.metric)

        # Create aligned spectrograms
        aligned_source = np.zeros_like(target_spec)
        aligned_target = np.zeros_like(target_spec)

        for i, j in path:
            aligned_source[j] = source_spec[i]
            aligned_target[j] = target_spec[j]

        return aligned_source, aligned_target

    def align_batch(self, source_specs, target_specs):
        """
        Align a batch of source spectrograms to target spectrograms.

        :param source_specs: list of source spectrograms
        :param target_specs: list of target spectrograms
        :return: list of tuples (aligned_source, aligned_target)
        """
        aligned_pairs = []
        for source, target in zip(source_specs, target_specs):
            aligned_source, aligned_target = self.align(source, target)
            aligned_pairs.append((aligned_source, aligned_target))
        return aligned_pairs


# Example usage
def preprocess_for_vae(spec):
    """
    Preprocess the spectrogram for input to the VAE.
    Assumes the VAE expects input of shape (batch_size, 1, height, width).
    """
    return np.expand_dims(spec, axis=(0, 1))


# Assuming you have your VAE model defined and trained
vae_model = VAE(input_dim=(128, 128), hidden_dim=500, latent_dim=20).to(device)
# vae_model.load_state_dict(torch.load('vae_model.pth'))
vae_model.eval()

aligner = SpectrogramAligner()


def transform_aligned_spectrograms(source_spec, target_spec, vae_model, aligner):
    # Align spectrograms
    aligned_source, aligned_target = aligner.align(source_spec, target_spec)

    # Preprocess for VAE
    source_input = torch.FloatTensor(preprocess_for_vae(aligned_source)).to(device)
    target_input = torch.FloatTensor(preprocess_for_vae(aligned_target)).to(device)

    with torch.no_grad():
        # Encode source and target
        source_mu, _ = vae_model.encode(source_input)
        target_mu, _ = vae_model.encode(target_input)

        # Interpolate in latent space
        interpolated_z = (source_mu + target_mu) / 2

        # Decode
        transformed_spec = vae_model.decode(interpolated_z)

    return transformed_spec.cpu().numpy().squeeze()


# Example transformation
# source_spec = ... # Your source spectrogram (numpy array)
# target_spec = ... # Your target spectrogram (numpy array)
# transformed_spec = transform_aligned_spectrograms(source_spec, target_spec, vae_model, aligner)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * (input_dim[0] // 8) * (input_dim[1] // 8), latent_dim)
        self.fc_logvar = nn.Linear(128 * (input_dim[0] // 8) * (input_dim[1] // 8), latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * (input_dim[0] // 8) * (input_dim[1] // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, self.input_dim[0] // 8, self.input_dim[1] // 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
def train(model, optimizer, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Average loss: {total_loss / len(dataloader.dataset):.4f}')

# Example usage
input_dim = (128, 128)  # Example spectrogram dimensions
hidden_dim = 500
latent_dim = 20
learning_rate = 1e-3
batch_size = 32
epochs = 50

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Assuming you have a DataLoader set up for your spectrograms
# train(model, optimizer, spectrogram_dataloader, epochs)

# To transform one spectrogram into another
def transform_spectrogram(model, source_spec, target_spec):
    model.eval()
    with torch.no_grad():
        # Encode source and target spectrograms
        source_mu, _ = model.encode(source_spec.unsqueeze(0))
        target_mu, _ = model.encode(target_spec.unsqueeze(0))

        # Interpolate in latent space
        interpolated_z = (source_mu + target_mu) / 2

        # Decode the interpolated latent vector
        transformed_spec = model.decode(interpolated_z)

    return transformed_spec.squeeze(0)import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * (input_dim[0] // 8) * (input_dim[1] // 8), latent_dim)
        self.fc_logvar = nn.Linear(128 * (input_dim[0] // 8) * (input_dim[1] // 8), latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * (input_dim[0] // 8) * (input_dim[1] // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, self.input_dim[0] // 8, self.input_dim[1] // 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
def train(model, optimizer, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Average loss: {total_loss / len(dataloader.dataset):.4f}')

# Example usage
input_dim = (128, 128)  # Example spectrogram dimensions
hidden_dim = 500
latent_dim = 20
learning_rate = 1e-3
batch_size = 32
epochs = 50

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Assuming you have a DataLoader set up for your spectrograms
# train(model, optimizer, spectrogram_dataloader, epochs)

# To transform one spectrogram into another
def transform_spectrogram(model, source_spec, target_spec):
    model.eval()
    with torch.no_grad():
        # Encode source and target spectrograms
        source_mu, _ = model.encode(source_spec.unsqueeze(0))
        target_mu, _ = model.encode(target_spec.unsqueeze(0))

        # Interpolate in latent space
        interpolated_z = (source_mu + target_mu) / 2

        # Decode the interpolated latent vector
        transformed_spec = model.decode(interpolated_z)

    return transformed_spec.squeeze(0)