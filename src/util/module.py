import torch
from transformers import pipeline

pipe = pipeline('image-classification', model='SM200203102097/skinDiseasesDetectionModel', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))