import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 1. Data Preparation

def load_text_data(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

data_dir = 'path_to_anime_stories'  # Replace with your data directory
texts = load_text_data(data_dir)
print(f"Loaded {len(texts)} stories.")

def build_vocab(texts, vocab_size=50000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(vocab_size - 2)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
        idx2word[idx] = word
    return word2idx, idx2word

vocab_size = 50000
word2idx, idx2word = build_vocab(texts, vocab_size=vocab_size)
print(f"Vocabulary size: {len(word2idx)}")

def encode_texts(texts, word2idx, max_length=1024):
    encoded_texts = []
    for text in texts:
        tokens = text.split()
        token_ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [word2idx['<PAD>']] * (max_length - len(token_ids))
        encoded_texts.append(token_ids)
    return np.array(encoded_texts)

max_length = 1024
encoded_texts = encode_texts(texts, word2idx, max_length=max_length)
print(f"Encoded texts shape: {encoded_texts.shape}")

split_ratio = 0.9
split_idx = int(len(encoded_texts) * split_ratio)
train_data = encoded_texts[:split_idx]
val_data = encoded_texts[split_idx:]

# 2. Model Definition

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.scale = self.d_k ** -0.5

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.out_proj(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=4 * 1024, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dropout=dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = x + attn_output
        x = self.ln1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.ln2(x)
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model=1024, num_layers=12, num_heads=12, max_len=1024, dropout=0.1):
        super(GPT2, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        for block in self.layers:
            nn.init.normal_(block.attn.q_linear.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attn.k_linear.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attn.v_linear.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.ffn.fc1.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.ffn.fc2.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        x = self.dropout(token_embeddings + position_embeddings)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# 3. Training Setup

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y

batch_size = 4  # Adjust as needed
train_dataset = TextDataset(train_data)
val_dataset = TextDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2(vocab_size=vocab_size, d_model=1024, num_layers=12, num_heads=12, max_len=1024, dropout=0.1)
model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 3
total_steps = num_epochs * len(train_loader)
warmup_steps = 500

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# 4. Model Training

scaler = torch.cuda.amp.GradScaler()
gradient_accumulation_steps = 4

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    optimizer.zero_grad()
    for step, (input_ids, targets) in enumerate(tqdm(train_loader)):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        epoch_loss += loss.item() * gradient_accumulation_steps

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 5. Inference and Evaluation

def generate_text(model, seed_text, max_length=100):
    model.eval()
    input_ids = [word2idx.get(token, word2idx['<UNK>']) for token in seed_text.split()]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=1)
    generated_text = ' '.join([idx2word[token.item()] for token in generated[0]])
    return generated_text

seed_text = "Once upon a time in an anime world"
start_time = time.time()
generated_text = generate_text(model, seed_text, max_length=150)
end_time = time.time()
inference_time = end_time - start_time
tokens_generated = len(generated_text.split())
tokens_per_sec = tokens_generated / inference_time
print(f"Inference Time: {inference_time:.2f} sec")
print(f"Tokens Generated: {tokens_generated}")
print(f"Tokens/sec: {tokens_per_sec:.2f}")
print(f"Generated Text:\n{generated_text}")

def calculate_perplexity(model, data_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, targets in tqdm(data_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * targets.ne(word2idx['<PAD>']).sum().item()
            total_tokens += targets.ne(word2idx['<PAD>']).sum().item()
    perplexity = np.exp(total_loss / total_tokens)
    return perplexity

val_perplexity = calculate_perplexity(model, val_loader)
print(f"Validation Perplexity: {val_perplexity:.2f}")
