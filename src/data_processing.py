import os
import random
import numpy as np
import librosa
import soundfile as sf
import shutil
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, sample_rate=22050, duration=5):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
        # Fixed spectrogram parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.fixed_length = 431  # Fixed time dimension
        
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, audio_path):
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pad or truncate to fixed length
        if len(waveform) > self.num_samples:
            waveform = waveform[:self.num_samples]
        elif len(waveform) < self.num_samples:
            waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))
        
        return waveform
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.load_audio(audio_path)
        
        # Convert to mel spectrogram with fixed parameters
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure fixed time dimension
        if mel_spec.shape[1] > self.fixed_length:
            mel_spec = mel_spec[:, :self.fixed_length]
        elif mel_spec.shape[1] < self.fixed_length:
            pad_width = self.fixed_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return torch.FloatTensor(mel_spec[None, :, :]), torch.tensor(self.labels[idx], dtype=torch.float32)

def augment_dataset(input_dir, output_dir):
    """
    Augment audio dataset by creating augmented versions of each file.
    
    Args:
        input_dir (str): Path to original dataset directory
        output_dir (str): Path to save augmented dataset
    """
    # Check if augmented dataset already exists
    if os.path.exists(output_dir):
        # Check if the augmented directories contain files
        capuchin_augmented = os.path.join(output_dir, "Parsed_Capuchinbird_Clips")
        not_capuchin_augmented = os.path.join(output_dir, "Parsed_Not_Capuchinbird_Clips")
        
        # If both directories exist and contain files, skip augmentation
        if (os.path.exists(capuchin_augmented) and 
            os.path.exists(not_capuchin_augmented) and 
            len(os.listdir(capuchin_augmented)) > 0 and 
            len(os.listdir(not_capuchin_augmented)) > 0):
            print("\tAugmented dataset already exists. Skipping augmentation.")
            return

    # If augmented dataset doesn't exist or is incomplete, create it
    # Create output directories
    os.makedirs(os.path.join(output_dir, "Parsed_Capuchinbird_Clips"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Parsed_Not_Capuchinbird_Clips"), exist_ok=True)
    
    # Process Capuchinbird Clips
    process_directory(
        os.path.join(input_dir, "Parsed_Capuchinbird_Clips"), 
        os.path.join(output_dir, "Parsed_Capuchinbird_Clips")
    )
    
    # Process Not Capuchinbird Clips
    process_directory(
        os.path.join(input_dir, "Parsed_Not_Capuchinbird_Clips"), 
        os.path.join(output_dir, "Parsed_Not_Capuchinbird_Clips")
    )

def process_directory(input_path, output_path):
    """
    Process and augment audio files in a directory.
    
    Args:
        input_path (str): Path to input directory
        output_path (str): Path to output directory
    """
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
    
    # Iterate through files in input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.wav'):
            # Full paths
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            augmented_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_augmented.wav")
            
            # Copy original file
            shutil.copy2(input_file, output_file)
            
            # Load audio
            waveform, sr = librosa.load(input_file, sr=22050)
            
            # Apply random augmentation
            augmentation_funcs = [pitch_shift, time_stretch, add_noise]     # Time shift wound up making results worse.
            aug_func = random.choice(augmentation_funcs)
            augmented_waveform = aug_func(waveform, sr)
            
            # Save augmented file
            sf.write(augmented_file, augmented_waveform, sr)