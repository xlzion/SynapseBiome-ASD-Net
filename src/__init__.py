"""
SynapseBiome ASD-Net: A Multimodal Deep Learning Framework for Autism Spectrum Disorder Diagnosis

This package provides a comprehensive framework for ASD diagnosis using fMRI and microbiome data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@institution.edu"

from .models import SynapseBiomeASDNet
from .data import ABIDE2DataProcessor
from .training import ContrastiveTrainer
from .analysis import BiomarkerAnalyzer

__all__ = [
    "SynapseBiomeASDNet",
    "ABIDE2DataProcessor", 
    "ContrastiveTrainer",
    "BiomarkerAnalyzer",
] 