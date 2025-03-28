import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256):
        super(SimpleDiffusionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Input spectra (batch_size, input_dim)
            t (torch.Tensor): Diffusion timestep (batch_size,)
        Returns:
            torch.Tensor: Denoised spectra
        """
        return self.network(x)

if __name__ == "__main__":
    # Test the model with simulated DIA-MS/MS data
    model = SimpleDiffusionModel()
    noisy_spectra = torch.randn(32, 100)  # Batch of 32 spectra, 100 features
    timestep = torch.ones(32)             # Dummy timestep
    denoised_spectra = model(noisy_spectra, timestep)
    print("Input shape:", noisy_spectra.shape)
    print("Output shape:", denoised_spectra.shape)