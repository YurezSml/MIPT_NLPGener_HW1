import os

import pickle
import dill

import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import model
from model import CrossEncoderBert

from tqdm import tqdm

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = 'data'

MODEL_FOLDER = 'model'

try:
  os.mkdir(MODEL_FOLDER)
except OSError as error:
  print(error)

def load_data(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
        return loaded_data

def prepare_dataset(dataset_file):
    dataset = load_data(os.path.join(DATA_FOLDER, dataset_file))

    print('Dataset')
    print(dataset[0])
    print(dataset[1])
    print(f'Lenght of data: {len(dataset)}')

    fixed_dataset = []
    for data in tqdm(dataset):
        curr_data = {}
        cont_quest = ''
        if len(data['context']) > 0:
            cont_quest = ''
            for pair in data['context']:
                cont_quest = str(pair[0]) + ' [U_token] '
                cont_quest = cont_quest + str(pair[1]) + ' [U_token] '
        curr_data['context_question'] = cont_quest + str(data['question'])
        curr_data['answer'] = str(data['answer'])
        curr_data['label'] = data['label']
        fixed_dataset.append(curr_data)

    return fixed_dataset


class DialogDataset(Dataset):
    def __init__(self, tokens: dict, labels: list[float]):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, ix: int) -> dict[str, torch.tensor]:
        return {
            "input_ids": torch.tensor(self.tokens["input_ids"][ix], dtype=torch.long),
            "attention_mask": torch.tensor(self.tokens["attention_mask"][ix], dtype=torch.long),
            "labels": torch.tensor(self.labels[ix], dtype=torch.float)
        }

    def __len__(self) -> int:
        return len(self.tokens["input_ids"])

def train_step_fn(model, optimizer, scheduler, loss_fn, batch):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    optimizer.zero_grad()
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def val_step_fn(model, loss_fn, batch):
    model.eval()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.squeeze(-1), labels)
    return loss.item()
    
def mini_batch(dataloader, step_fn, dataset_size, batch_size, is_training=True):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_steps = dataset_size // batch_size

    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps - warmup_steps)

    loss_fn = torch.nn.MSELoss()

    mini_batch_losses = []
    for i, batch in enumerate(dataloader):
        if is_training:
            loss = step_fn(model, optimizer, scheduler, loss_fn, batch)
        else:
            loss = step_fn(model, loss_fn, batch)
        mini_batch_losses.append(loss)
        if i % (batch_size * 4) == 0:
            print(f"Step {i:>5}/{len(dataloader)}, Loss = {loss:.3f}")
    return np.mean(mini_batch_losses), mini_batch_losses

def start_training(dataset, model, n_epochs = 1):

    train_val_ratio = 0.8

    n_total = len(dataset)
    n_train = int(n_total * train_val_ratio)

    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'train_length: {len(train_dataloader)}, val_length: {len(val_dataloader)}')

    train_losses, train_mini_batch_losses = [], []
    val_losses, val_mini_batch_losses = [], []

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}")
        train_loss, _train_mini_batch_losses = mini_batch(train_dataloader, train_step_fn, len(train_dataset), batch_size, is_training=True)
        train_mini_batch_losses.extend(_train_mini_batch_losses)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss, _val_mini_batch_losses = mini_batch(val_dataloader, val_step_fn, len(train_dataset), batch_size, is_training=False)
            val_mini_batch_losses.extend(_val_mini_batch_losses)
            val_losses.append(val_loss)

    window_size = 32

    train_mb_running_loss = []
    for i in range(len(train_mini_batch_losses)-window_size):
        train_mb_running_loss.append(np.mean(train_mini_batch_losses[i:i+window_size]))

    val_mb_running_loss = []
    for i in range(len(val_mini_batch_losses)-window_size):
        val_mb_running_loss.append(np.mean(val_mini_batch_losses[i:i+window_size]))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(range(len(train_mb_running_loss)), train_mb_running_loss);
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    plt.savefig(os.path.join(MODEL_FOLDER, 'train_loss.png'))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(range(len(val_mb_running_loss)), val_mb_running_loss);
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    plt.savefig(os.path.join(MODEL_FOLDER, 'val_loss.png'))

    return model


if __name__ == "__main__":

    fixed_dataset = prepare_dataset('data.pkl')

    print('Fixed dataset')
    print(fixed_dataset[0])
    print(fixed_dataset[1])

    text_lengths1, text_lengths2 = [], []
    for data in tqdm(fixed_dataset):
        text_lengths1.append(len(data['context_question']))
        text_lengths2.append(len(data['answer']))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].hist(text_lengths1)
    ax[0].set_xlabel('Длина вопроса')
    ax[0].set_ylabel('Количество вопросов')
    ax[1].hist(text_lengths2)
    ax[1].set_xlabel('Длина ответа')
    ax[1].set_ylabel('Количество ответов')
    plt.savefig(os.path.join(DATA_FOLDER, 'text_lengths.png'))

    # Из анализа распределения длин устанавливаем максимальную длину для tokenizer
    MAX_LENGTH = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoderBert().to(device)
    model.model.resize_token_embeddings(len(model.tokenizer))

    tokenized_texts = model.tokenizer([data['context_question'] for data in fixed_dataset],
                                    [data['answer'] for data in fixed_dataset],
                                    max_length=MAX_LENGTH, padding="max_length",
                                    truncation=True, verbose=True)

    dataset = DialogDataset(tokenized_texts, [data["label"] for data in fixed_dataset])

    model = start_training(dataset, model, 1)

    # Сохраняем модель и токенайзер с дополнительным токеном [U_token]

    serialized = dill.dumps(model)
    with open(os.path.join(MODEL_FOLDER, 'friends_model.pkl'), 'wb') as output:
        output.write(serialized)

    model.tokenizer.save_pretrained(os.path.join(MODEL_FOLDER, 'tokenizer'))