from .base import Dataset, DataLoader
from .loaders import BreastCancerLoader, HeartDiseaseLoader, DiabetesLoader

DATA_LOADERS = {
    "breast_cancer": BreastCancerLoader,
    "heart_disease": HeartDiseaseLoader,
    "diabetes": DiabetesLoader,
}

__all__ = ["Dataset", "DataLoader", "DATA_LOADERS"]
