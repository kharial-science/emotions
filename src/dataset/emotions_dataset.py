import re
import nltk
from nltk.corpus import stopwords

import torch
import pandas as pd
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext import transforms as T

import config

nltk.data.path.append("..\\data\\nltk")


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
        self.vocab = self.build_vocab()
        self.emotions = list(self.data["Emotion"].unique())

    def __len__(self):
        return len(self.data)

    def build_vocab(self):
        """
        Builds a vocabulary from the dataset.

        Returns:
            torchtext.vocab.Vocab: The vocabulary object.
        """

        vocab = build_vocab_from_iterator(
            self.data["Text"]
            .apply(self.clean_text)
            .apply(self.remove_stop_words)
            .apply(self.tokenize)
            .apply(self.lemmatize),
            min_freq=2,
            specials=["<unk>", "<pad>"],
            special_first=True,
            max_tokens=config.MAX_WORDS,
        )

        vocab.set_default_index(vocab["<unk>"])

        return vocab

    def clean_text(self, text: str, max_length: int = None):
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
        # Remove newline characters
        text = text.replace("\n", "")
        # Convert to lowercase
        text = text.lower()

        if max_length is not None:
            text = text[:max_length]

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

    def tokenize(self, text: str, char_based: bool = False, fix_length: int = None):
        """
        Tokenizes the given text into a list of tokens.

        Args:
            text (str): The input text to be tokenized.
            char_based (bool, optional): If True, tokenizes the text into individual characters.
                If False (default), tokenizes the text into words using space as the delimiter.
            fix_length (int, optional): The maximum length of the tokenized list. If specified,
                the list will be padded or truncated to this length.

        Returns:
            list: A list of tokens.
        """

        if char_based:
            tokens = list(text)
        else:
            tokens = text.split(" ")

        if fix_length is not None:
            if len(tokens) < fix_length:
                tokens = ["<pad>"] * (fix_length - len(tokens)) + tokens
            else:
                tokens = tokens[:fix_length]

        return tokens

    def lemmatize(self, text: [str]):
        return text  # TODO: Implement

    def turn_to_indices(self, text: [str], vocab):
        """
        Turns the given text into a list of indices based on the given vocabulary.

        Args:
            text (list): The text to be converted.
            vocab (torchtext.vocab.Vocab): The vocabulary to use for conversion.

        Returns:
            list: A list of indices.
        """
        text_transform = T.Sequential(
            T.VocabTransform(vocab),
            T.ToTensor(),
        )
        return text_transform(text)

    def encode_emotion(self, emotion: int, num_emotions: int):
        """
        Encodes the given emotion as a one-hot vector.

        Args:
            emotion (int): The index of the emotion to encode.
            num_emotions (int): The total number of emotions.

        Returns:
            torch.Tensor: The one-hot encoded emotion vector.
        """

        emotion_one_hot = torch.zeros(num_emotions)  # , dtype=torch.long
        emotion_one_hot[emotion] = 1
        return emotion_one_hot

    def process_text(self, text: str):
        """
        Processes the given text by cleaning, tokenizing, lemmatizing, and turning it into indices.

        Args:
            text (str): The text to be processed.

        Returns:
            torch.Tensor: The processed text.
        """
        text = self.clean_text(text)
        text = self.remove_stop_words(text)
        text = self.tokenize(text, char_based=False, fix_length=config.WORDS_PER_SAMPLE)
        text = self.lemmatize(text)
        text = self.turn_to_indices(text, self.vocab)
        return text

    def __getitem__(self, idx):
        # Implement your own logic to retrieve data from the dataset
        sample = self.data.iloc[idx]

        # Clean the text
        text = self.process_text(sample["Text"])

        # Encode emotion to one-hot vector
        emotion = sample["Emotion"]
        emotion = list(self.emotions).index(emotion)
        emotion_one_hot = self.encode_emotion(emotion, config.NUM_EMOTIONS)

        # Return the sample as a dictionary or tuple
        return {
            "text": text,
            "emotion": emotion_one_hot,
            # "emotion_text": sample["Emotion"],
        }
