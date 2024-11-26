import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def generate_demo():
    # Find the first capuchinbird clip
    capuchin_dir = Path("data/Parsed_Capuchinbird_Clips")
    example_files = list(capuchin_dir.glob("*.wav"))
    
    if not example_files:
        print("No capuchinbird clips found!")
        return
    
    example_wav = example_files[0]
    print(f"Using example file: {example_wav.name}")
    
    # Load audio
    waveform, sr = librosa.load(example_wav)
    
    # Original Waveform and Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(waveform, sr=sr)
    plt.title("Original Waveform")
    plt.tight_layout()
    plt.savefig("Figures/Waveform.png")
    plt.close()
    
    plt.figure(figsize=(10, 4))
    D = librosa.stft(waveform)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                              sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Spectrogram")
    plt.tight_layout()
    plt.savefig("Figures/Spectrogram.png")
    plt.close()
    
    # Augmentation functions
    def time_shift(y, sr):
        shift_factor = random.uniform(-0.2, 0.2)
        shift_samples = int(len(y) * shift_factor)
        return np.roll(y, shift_samples)
    
    def pitch_shift(y, sr):
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    def time_stretch(y, sr):
        stretch_factor = random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y=y, rate=stretch_factor)
    
    def add_noise(y, sr):
        noise_factor=0.005
        noise = np.random.normal(0, noise_factor, len(y))
        return y + noise
    
    # Individual augmentations
    augmentations = [
        ('TimeShifted', time_shift),
        ('PitchShifted', pitch_shift),
        ('TimeStretched', time_stretch),
        ('NoiseAdded', add_noise)
    ]
    
    for name, aug_func in augmentations:
        # Apply augmentation
        aug_waveform = aug_func(waveform, sr)
        
        # Waveform plot
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(aug_waveform, sr=sr)
        plt.title(f"{name} Waveform")
        plt.tight_layout()
        plt.savefig(f"Figures/Waveform{name}.png")
        plt.close()
        
        # Spectrogram plot
        plt.figure(figsize=(10, 4))
        D = librosa.stft(aug_waveform)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                                  sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{name} Spectrogram")
        plt.tight_layout()
        plt.savefig(f"Figures/Spectrogram{name}.png")
        plt.close()
    
    # Combined augmentations
    combined_aug_waveform = time_stretch(
        pitch_shift(
            time_shift(
                add_noise(waveform, sr), sr), sr), sr)
    
    # Combined Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(combined_aug_waveform, sr=sr)
    plt.title("Combined Augmentations Waveform")
    plt.tight_layout()
    plt.savefig("Figures/WaveformAugmented.png")
    plt.close()
    
    # Combined Spectrogram plot
    plt.figure(figsize=(10, 4))
    D = librosa.stft(combined_aug_waveform)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                              sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Combined Augmentations Spectrogram")
    plt.tight_layout()
    plt.savefig("Figures/SpectrogramAugmented.png")
    plt.close()



    # Extreme Augmentation functions
    def extreme_time_shift(y, sr):
        shift_factor = random.uniform(-0.6, 0.6)
        shift_samples = int(len(y) * shift_factor)
        return np.roll(y, shift_samples)
    
    def extreme_pitch_shift(y, sr):
        n_steps = random.uniform(-6, 6)
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    def extreme_time_stretch(y, sr):
        stretch_factor = random.uniform(0.4, 1.6)
        return librosa.effects.time_stretch(y=y, rate=stretch_factor)
    
    def extreme_add_noise(y, sr):
        noise_factor=0.015
        noise = np.random.normal(0, noise_factor, len(y))
        return y + noise
    
    # Individual extreme augmentations
    extreme_augmentations = [
        ('ExtremeTimeShifted', extreme_time_shift),
        ('ExtremePitchShifted', extreme_pitch_shift),
        ('ExteremeTimeStretched', extreme_time_stretch),
        ('ExteremeNoiseAdded', extreme_add_noise)
    ]
    
    for name, aug_func in extreme_augmentations:
        # Apply augmentation
        aug_waveform = aug_func(waveform, sr)
        
        # Waveform plot
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(aug_waveform, sr=sr)
        plt.title(f"{name} Waveform")
        plt.tight_layout()
        plt.savefig(f"Figures/Waveform{name}.png")
        plt.close()
        
        # Spectrogram plot
        plt.figure(figsize=(10, 4))
        D = librosa.stft(aug_waveform)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                                  sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{name} Spectrogram")
        plt.tight_layout()
        plt.savefig(f"Figures/Spectrogram{name}.png")
        plt.close()
    
    # Combined augmentations
    combined_aug_waveform = extreme_time_stretch(
        extreme_pitch_shift(
            extreme_time_shift(
                extreme_add_noise(waveform, sr), sr), sr), sr)
    
    # Combined Waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(combined_aug_waveform, sr=sr)
    plt.title("Combined Augmentations Waveform")
    plt.tight_layout()
    plt.savefig("Figures/WaveformExtremeAugmented.png")
    plt.close()
    
    # Combined Spectrogram plot
    plt.figure(figsize=(10, 4))
    D = librosa.stft(combined_aug_waveform)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                              sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Combined Augmentations Spectrogram")
    plt.tight_layout()
    plt.savefig("Figures/SpectrogramExtremeAugmented.png")
    plt.close()

    print("Visualization and augmentation demo complete. Check the generated PNG files.")
