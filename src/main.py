import torch
from torch.utils.data import DataLoader
from src import config

from src.dataset.emotions_dataset import EmotionsDataset

# Create a dataset and dataloader
dataset = EmotionsDataset("../data/emotions-in-text/Emotion_final.csv")
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
