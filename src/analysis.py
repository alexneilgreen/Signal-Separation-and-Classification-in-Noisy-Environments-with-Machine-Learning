import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

from .model import AudioCNN

class AudioTarget:
    def __init__(self, start_time, end_time=None, confidence=None):
        self.start_time = start_time
        self.end_time = end_time if end_time is not None else start_time
        self.confidences = [confidence] if confidence is not None else []
    
    def extend_call(self, time, confidence):
        self.end_time = time
        self.confidences.append(confidence)
    
    @property
    def duration(self):
        return self.end_time - self.start_time
    
    @property
    def average_confidence(self):
        return sum(self.confidences) / len(self.confidences) if self.confidences else None

class ContinuousAudioAnalyzer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AudioCNN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.eval()
        
        # Audio parameters (matching training settings)
        self.sample_rate = 22050
        self.duration = 5  # seconds per segment
        self.hop_length = 2.5  # seconds to slide window
        self.n_fft = 2048
        self.hop_length_spec = 512
        self.n_mels = 128
        self.fixed_length = 431
        
        # Parameters for call grouping
        self.min_gap = 3.0  # minimum gap in seconds to consider calls separate
        self.confidence_threshold = 0.8  # threshold for detection
    
    def preprocess_audio_segment(self, waveform):
        """Convert audio segment to mel spectrogram with same parameters as training."""
        # Ensure correct length
        target_length = int(self.sample_rate * self.duration)
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[:target_length]
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length_spec
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
        
        return torch.FloatTensor(mel_spec[None, None, :, :])
    
    def group_detections(self, detections):
        """Group consecutive detections into single bird calls."""
        if not detections:
            return []
        
        grouped_calls = []
        current_call = None
        
        for time, confidence in sorted(detections):
            if current_call is None:
                current_call = AudioTarget(time, confidence=confidence)
            elif time - current_call.end_time > self.min_gap:
                if current_call.duration > 0:                       # Only append the current call if it has non-zero duration
                    grouped_calls.append(current_call)
                current_call = AudioTarget(time, confidence=confidence)
            else:
                current_call.extend_call(time, confidence)
        
        if current_call is not None and current_call.duration > 0:  # Check the last call
            grouped_calls.append(current_call)
        
        return grouped_calls
    
    def analyze_file(self, audio_path):
        """Analyze a single audio file and return detected bird calls."""
        # Load audio file
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate number of segments
        hop_length_samples = int(self.hop_length * self.sample_rate)
        segment_samples = int(self.duration * self.sample_rate)
        num_segments = int(np.ceil((len(waveform) - segment_samples) / hop_length_samples)) + 1
        
        detections = []
        
        # Process each segment
        for i in range(num_segments):
            start_sample = i * hop_length_samples
            end_sample = start_sample + segment_samples
            segment = waveform[start_sample:end_sample]
            
            # Skip if segment is too short
            if len(segment) < self.sample_rate:
                continue
            
            # Preprocess segment
            features = self.preprocess_audio_segment(segment)
            features = features.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(features)
                confidence = output.item()
                
                if confidence > self.confidence_threshold:
                    time_in_seconds = start_sample / self.sample_rate
                    detections.append((time_in_seconds, confidence))
        
        # Group consecutive detections
        grouped_calls = self.group_detections(detections)
        
        return grouped_calls
    
    def analyze_directory(self, directory_path, output_file='Results/Results.csv'):
        """Analyze all audio files in directory and save detailed results to CSV."""
        directory = Path(directory_path)
        results = []
        all_files = {}  # Dictionary to track all files and their results
        
        # Get all audio files
        audio_files = list(directory.glob('*.mp3')) + list(directory.glob('*.wav'))
        
        # Initialize all files with zero counts
        for audio_file in audio_files:
            all_files[audio_file.name] = {'Total_Calls': 0}
        
        print(f"Found {len(audio_files)} audio files to analyze")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc="Analyzing recordings"):
            try:
                calls = self.analyze_file(audio_file)
                
                # Create detailed entry for each call
                for i, call in enumerate(calls, 1):
                    results.append({
                        'Filename': audio_file.name,
                        'Call_Number': i,
                        'Start_Time': f"{call.start_time:.2f}",
                        'End_Time': f"{call.end_time:.2f}",
                        'Duration': f"{call.duration:.2f}",
                        'Average_Confidence': f"{call.average_confidence:.3f}"
                    })
                
                # Update file summary if calls were found
                if calls:
                    all_files[audio_file.name] = {'Total_Calls': len(calls)}
            
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")
        
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        # Create summary DataFrame including files with zero calls
        summary_df = pd.DataFrame.from_dict(all_files, orient='index')
        summary_df.index.name = 'Filename'
        
        summary_file = output_file.replace('.csv', 'Summary.csv')
        summary_df.to_csv(summary_file)
        
        print(f"Detailed results saved to {output_file}")
        print(f"Summary results saved to {summary_file}")

def compare_results(results_path, ground_truth_path):
    """
    Compare predicted results with ground truth and create visualization.
    
    Args:
        results_path (str): Path to ResultsSummary.csv
        ground_truth_path (str): Path to GroundTruth.csv
    """
    # Read the CSV files
    results_df = pd.read_csv(results_path)
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    # Merge the dataframes on Filename
    combined_df = pd.merge(results_df, ground_truth_df, 
                          on='Filename', 
                          suffixes=('_predicted', '_truth'))
    
    # Calculate metrics
    mae = mean_absolute_error(combined_df['Total_Calls_truth'], 
                            combined_df['Total_Calls_predicted'])
    mse = mean_squared_error(combined_df['Total_Calls_truth'], 
                           combined_df['Total_Calls_predicted'])
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_df['Total_Calls_truth'], 
                  combined_df['Total_Calls_predicted'])
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Scatter plot of predictions vs ground truth
    plt.scatter(combined_df['Total_Calls_truth'], 
               combined_df['Total_Calls_predicted'], 
               alpha=0.5, 
               color='blue', 
               label='Predictions')
    
    # Perfect prediction line
    max_calls = max(max(combined_df['Total_Calls_truth'].max(), 
                       combined_df['Total_Calls_predicted'].max()),
                   1)  # Ensure at least 1 for scale
    plt.plot([0, max_calls], [0, max_calls], 
             'r--', 
             label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Ground Truth (Number of Calls)')
    plt.ylabel('Model Predictions (Number of Calls)')
    plt.title('Model Predictions vs Ground Truth')
    
    # Add metrics text box
    metrics_text = (f'Metrics:\n'
                   f'MAE: {mae:.2f}\n'
                   f'RMSE: {rmse:.2f}\n'
                   f'R²: {r2:.3f}')
    
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontfamily='monospace')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Force axes to start at 0,0 and maintain equal scale
    plt.xlim(0, max_calls * 1.05)  # Add 5% padding
    plt.ylim(0, max_calls * 1.05)
    plt.axis('equal')
    
    # Ensure Results directory exists
    results_dir = Path('Results')
    results_dir.mkdir(exist_ok=True)
    
    # Save the figure in the Results folder
    plt.savefig(results_dir / 'ModelResultsAccuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return mae, rmse, r2

def check_and_compare_results(results_path, ground_truth_path):
    """
    Check if files exist before comparing results.
    
    Args:
        results_path (str): Path to ResultsSummary.csv
        ground_truth_path (str): Path to GroundTruth.csv
        
    Returns:
        tuple or None: (mae, rmse, r2) if successful, None if files missing
    """
    # Convert to Path objects
    results_file = Path(results_path)
    ground_truth_file = Path(ground_truth_path)
    
    # Check if both files exist
    missing_files = []
    
    if not results_file.exists():
        missing_files.append(results_path)
    
    if not ground_truth_file.exists():
        missing_files.append(ground_truth_path)
    
    # If any files are missing, print error message and return
    if missing_files:
        print("Error: The following files were not found:")
        for file in missing_files:
            print(f"- {file}")
        return None
    
    # If both files exist, proceed with comparison
    print("Files found. Proceeding with comparison...")
    return compare_results(results_path, ground_truth_path)