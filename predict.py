from cog import BasePredictor, BaseModel, Path, File
import torch
import os
import wget
import librosa
import numpy as np
from pathlib import Path as PathLib
import json

from inference.rosvot import RosvotInfer
from utils.commons.hparams import set_hparams

class Output(BaseModel):
    midi_file: File
    notes: File  # For the .npy file containing note information

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Create necessary directories
        os.makedirs("checkpoints/rosvot", exist_ok=True)
        os.makedirs("checkpoints/rwbd", exist_ok=True)
        os.makedirs("checkpoints/rmvpe", exist_ok=True)
        
        # URLs from repository analysis
        # Note: Based on README.md, models are hosted on Google Drive
        # Using direct download link from Google Drive
        model_urls = {
            'rosvot': "https://drive.google.com/uc?id=1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh",
            'rwbd': "https://drive.google.com/uc?id=YOUR_RWBD_ID",  # Need actual ID
            'rmvpe': "https://drive.google.com/uc?id=YOUR_RMVPE_ID"  # Need actual ID
        }
        
        # Download model files if they don't exist
        for model_name, url in model_urls.items():
            model_path = f"checkpoints/{model_name}/model.pt"
            if not os.path.exists(model_path):
                print(f"Downloading {model_name} model...")
                wget.download(url, model_path)
        
        # Initialize ROSVOT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rosvot = RosvotInfer(num_gpus=1)  # Using single GPU mode for inference
    
    def predict(
        self,
        audio_file: Path,
        threshold: float = 0.85,
        max_frames: int = 30000,
        save_plot: bool = True,
    ) -> Output:
        """Run a single prediction on the model"""
        try:
            # Create temporary output directory
            output_dir = PathLib("output")
            output_dir.mkdir(exist_ok=True)
            
            # Prepare metadata for single file processing
            metadata = [{
                "item_name": "output",
                "wav_fn": str(audio_file)
            }]
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Set up arguments for ROSVOT
            args = [
                "-o", str(output_dir),
                "--metadata", str(metadata_file),
                "--thr", str(threshold),
                "--max_frames", str(max_frames)
            ]
            if save_plot:
                args.append("--save_plot")
            
            # Run inference
            results = self.rosvot.run()
            
            # Get output files
            midi_file = output_dir / "midi" / "output.mid"
            notes_file = output_dir / "npy" / "[note]output.npy"
            
            if not midi_file.exists() or not notes_file.exists():
                raise RuntimeError("Failed to generate output files")
            
            return Output(
                midi_file=File(midi_file),
                notes=File(notes_file)
            )
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
