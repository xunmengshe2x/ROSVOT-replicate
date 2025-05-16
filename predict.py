from cog import BasePredictor, Input, Output, Path
import torch
import os
import wget
import librosa
import numpy as np
from pathlib import Path as PathLib

from inference.rosvot import RosvotInfer
from utils.commons.hparams import set_hparams

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Download model checkpoints
        # URL from README: https://drive.google.com/file/d/1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh/view?usp=sharing
        checkpoint_url = "https://drive.google.com/uc?export=download&id=1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh"
        if not os.path.exists("checkpoints/checkpoints.zip"):
            print("Downloading model checkpoints...")
            wget.download(checkpoint_url, "checkpoints/checkpoints.zip")
            os.system("cd checkpoints && unzip checkpoints.zip")

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RosvotInfer(num_gpus=1)

    def predict(
        self,
        audio: Path = Input(description="Input audio file"),
        threshold: float = Input(
            description="Threshold to determine note boundaries",
            default=0.85,
            ge=0.0,
            le=1.0,
        ),
        save_plot: bool = Input(
            description="Whether to save visualization plots",
            default=True
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Create output directory
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Set up arguments
        self.model.args.wav_fn = str(audio)
        self.model.args.save_dir = output_dir
        self.model.args.thr = threshold
        self.model.args.save_plot = save_plot
        
        # Run inference
        results = self.model.run()

        # Prepare outputs
        midi_path = PathLib(output_dir) / "midi" / "output.mid"
        plot_path = PathLib(output_dir) / "plot" / "[MIDI]output.png"
        
        return Output(
            midi_file=Path(midi_path),
            visualization=Path(plot_path) if save_plot else None,
            results=results
        )