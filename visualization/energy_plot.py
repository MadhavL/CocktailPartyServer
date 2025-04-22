import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_spatial_energy(exp, noisy_stft: torch.Tensor, 
                       angle_resolution: int = 2,
                       reference_angle: Optional[float] = None,
                       title: str = "Spatial Energy Distribution"):
    """
    Creates a polar plot showing audio energy distribution around the microphone array.
    
    Args:
        exp: The experiment object containing the model
        noisy_stft: The STFT of the noisy signal [BATCH, CHANNELS, FREQ, TIME]
        angle_resolution: Angular resolution in degrees
        reference_angle: Optional angle to highlight (e.g., known speaker direction)
        title: Plot title
    """
    # Prepare input
    stacked_noisy_stft = torch.concat(
        (torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1
    )
    
    # Scan all angles
    angles = np.arange(-180, 180, angle_resolution)
    energies = []
    
    for angle in angles:
        # Encode angle condition
        angle_enc = exp.encode_condition(
            torch.tensor([angle], dtype=torch.long, device=exp.device)
        )
        
        # Get mask for this angle
        stacked_mask = exp.model(
            stacked_noisy_stft, angle_enc, device=exp.device
        )
        mask, _ = exp.get_complex_masks_from_stacked(stacked_mask)
        
        # Apply mask and compute energy
        enhanced_stft = noisy_stft[:, exp.reference_channel, ...] * mask
        energy = torch.mean(torch.abs(enhanced_stft) ** 2).cpu().item()
        energies.append(energy)
    
    # Convert to numpy and normalize
    energies = np.array(energies)
    energies = energies / np.max(energies)
    
    # Create polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Convert angles to radians for plotting
    theta = np.radians(angles)
    
    # Plot energy distribution
    ax.plot(theta, energies)
    ax.fill(theta, energies, alpha=0.25)
    
    # Add reference angle if provided
    if reference_angle is not None:
        ref_theta = np.radians(reference_angle)
        ax.plot([ref_theta, ref_theta], [0, 1], 'r--', label='Reference Direction')
        
    # Customize plot
    ax.set_theta_zero_location("N")  # 0 degrees at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_title(title)
    ax.grid(True)
    
    if reference_angle is not None:
        ax.legend()
    
    return fig 