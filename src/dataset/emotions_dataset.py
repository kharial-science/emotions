import re
import torch
import pandas as pd
from torch.utils.data import Dataset
import nltk
from nltk.corpus import stopwords


class EmotionsDataset(Dataset):
    """
    A custom dataset class for emotions data.

    Args:
        csv_file (str): The path to the CSV file containing the dataset.

    Attributes:
        data (pandas.DataFrame): The loaded dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        clean_text(text: str, max_length: int): Cleans the text by removing punctuation, sequences of spaces, fixing the length, and removing newline characters.
        tokenize(text: str, char_based: bool = False): Tokenizes the text into a list of words or characters.
        lemmatize(text: List[str]): Lemmatizes the text (not implemented).
        encode_emotion(emotion: int, num_emotions: int): Encodes the emotion as a one-hot vector.
        __getitem__(idx): Retrieves a sample from the dataset.

    """

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def clean_text(self, text: str, max_length: int):
        """
        Cleans the given text by removing punctuation, sequences of spaces, newline characters,
        and fixing the length to the specified maximum length.

        Args:
            text (str): The text to be cleaned.
            max_length (int): The maximum length of the cleaned text.

        Returns:
            str: The cleaned text.
        """

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove sequences of spaces
        text = re.sub(r"\s+", " ", text)
        # Fix the length
        text = text[:max_length].ljust(max_length)
        # Remove newline characters
        text = text.replace("\n", "")
        # Convert to lowercase
        text = text.lower()

        return text

    def remove_stop_words(self, text: str):
        """
        Removes stop words from the given text.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        stop_words = set(stopwords.words("english"))
        tokens = nltk.word_tokenize(text)
        filtered_text = [word for word in tokens if word.lower() not in stop_words]
        cleaned_text = " ".join(filtered_text)
        return cleaned_text

    def tokenize(self, text: str, char_based: bool = False):
        """
        Tokenizes the given text into a list of tokens.

        Args:
            text (str): The input text to be tokenized.
            char_based (bool, optional): If True, tokenizes the text into individual characters.
                If False (default), tokenizes the text into words using space as the delimiter.

        Returns:
            list: A list of tokens.
        """

        if char_based:
            return list(text)
        return text.split(" ")

    def lemmatize(self, text: [str]):
        return text  # TODO: Implement

    def encode_emotion(self, emotion: int, num_emotions: int):
        """
        Encodes the given emotion as a one-hot vector.

        Args:
            emotion (int): The index of the emotion to encode.
            num_emotions (int): The total number of emotions.

        Returns:
            torch.Tensor: The one-hot encoded emotion vector.
        """

        emotion_one_hot = torch.zeros(num_emotions)
        emotion_one_hot[emotion] = 1
        return emotion_one_hot

    def __getitem__(self, idx):
        # Implement your own logic to retrieve data from the dataset
        sample = self.data.iloc[idx]

        # Clean the text
        text = self.clean_text(sample["Text"], max_length=100)
        text = self.remove_stop_words(text)
        text = self.tokenize(text, char_based=False)
        text = self.lemmatize(text)

        # Encode emotion to one-hot vector
        emotions = self.data["Emotion"].unique()
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
