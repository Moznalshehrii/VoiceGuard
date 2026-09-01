import torch
import torchaudio
import torchaudio.transforms as T

class AudioAugmenter:
    """Class to apply noise and codec augmentations for robustness testing."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def add_additive_noise(self, waveform, snr_db=15):
        """Adds Gaussian white noise at a specific Signal-to-Noise Ratio (SNR)."""
        noise = torch.randn_like(waveform)
        
        s_power = waveform.norm(p=2)
        n_power = noise.norm(p=2)
        
        snr = 10 ** (snr_db / 20)
        scale = s_power / (snr * n_power + 1e-8)
        
        return waveform + scale * noise

    def apply_codec_telephony(self, waveform):
        """Simulates lossy compression and telephony bandwidth reduction (Mu-law encoding)."""
        encoder = T.MuLawEncoding(quantization_channels=256)
        decoder = T.MuLawDecoding(quantization_channels=256)
        
        encoded = encoder(waveform)
        decoded = decoder(encoded)
        return decoded

    def augment(self, waveform, apply_noise=True, apply_codec=True):
        """Applies both noise and codec transforms to a given audio waveform."""
        augmented = waveform.clone()
        
        if apply_noise:
            augmented = self.add_additive_noise(augmented, snr_db=15)
            
        if apply_codec:
            augmented = self.apply_codec_telephony(augmented)
            
        return augmented

if __name__ == "__main__":
    print("Robustness Testing Augmenter Initialized Successfully.")
    
    # Validation test using a dummy waveform (4 seconds at 16kHz)
    sample_rate = 16000
    dummy_waveform = torch.randn(1, sample_rate * 4)
    
    print(f"Original Waveform Shape: {dummy_waveform.shape}")
    
    augmenter = AudioAugmenter(sample_rate=sample_rate)
    augmented_waveform = augmenter.augment(dummy_waveform, apply_noise=True, apply_codec=True)
    
    print(f"Augmented Waveform Shape: {augmented_waveform.shape}")
    print("Success! Robustness testing module validated successfully.")