import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import config
from dataset.emotions_dataset import EmotionsDataset

# Create a dataset
dataset = EmotionsDataset("../data/emotions-in-text/Emotion_final.csv")

# Split the dataset into train and eval
train_size = int(len(dataset) * config.TRAIN_RATIO)
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

dataset.build_vocab()

# Create dataloaders for train and eval
train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


class EmotionsClassifier(nn.Module):
    def __init__(self, num_words, embedding_dim, num_emotions):
        super().__init__()

        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            100,
            # bidirectional=True,
            batch_first=True,
            # dropout=0.2
        )
        self.linear = nn.Linear(100, num_emotions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.linear(x[1][0][0])
        # x[0] is the output of the LSTM, x[1] is the hidden state of the LSTM,
        # contains the last hidden state and last cell state
        x = self.softmax(x)
        return x


model = EmotionsClassifier(config.MAX_WORDS, config.EMBEDDING_DIM, config.NUM_EMOTIONS)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, verbose=True
)

for epoch in range(config.EPOCHS):
    # Train the model
    model.train()
    i = 0
    running_loss = 0.0
    for batch in train_dataloader:
        # Get the inputs and labels
        inputs = batch["text"]
        labels = batch["emotion"]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
            running_loss = 0.0
        i += 1

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            # Get the inputs and labels
            inputs = batch["text"]
            labels = batch["emotion"]

            # Predict and count
            outputs = model(inputs)
            total += labels.size(0)
            correct += (
                (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1))
                .sum()
                .item()
            )

    scheduler.step(correct / total)

    print(f"Accuracy: {100 * correct / total}%")
