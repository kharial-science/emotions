import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re
import torch
import torch.nn.functional as F


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def clean_text(self, text: str, max_length: int):
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove sequences of spaces
        text = re.sub(r"\s+", " ", text)
        # Fix the length
        text = text[:max_length].ljust(max_length)
        # Remove newline characters
        text = text.replace("\n", "")

        return text

    def tokenize(self, text: str, char_based: bool = False):
        if char_based:
            return list(text)
        return text.split(" ")

    def lemmatize(self, text: [str]):
        return text  # TODO: Implement

    def encode_emotion(self, emotion: int, num_emotions: int):
        emotion_one_hot = torch.zeros(num_emotions)
        emotion_one_hot[emotion] = 1
        return emotion_one_hot

    def __getitem__(self, idx):
        # Implement your own logic to retrieve data from the dataset
        sample = self.data.iloc[idx]

        # Clean the text
        text = self.clean_text(sample["Text"], max_length=100)
        text = self.tokenize(text, char_based=False)
        text = self.lemmatize(text)

        # Encode emotion to one-hot vector
        emotions = dataset.data["Emotion"].unique()
        num_emotions = len(emotions)
        emotion = sample["Emotion"]
        emotion = list(emotions).index(emotion)
        emotion_one_hot = self.encode_emotion(emotion, num_emotions)

        # Return the sample as a dictionary or tuple
        return {
            "text": text,
            "emotion": emotion_one_hot,
            "emotion_text": sample["Emotion"],
        }


# Create an instance of your custom dataset
dataset = MyDataset("../data/emotions-in-text/Emotion_final.csv")

print(dataset[3])

# Create a DataLoader from the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
