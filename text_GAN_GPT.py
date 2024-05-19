import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Датасет
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > block_size:
                for i in range(0, len(tokens) - block_size):
                    self.examples.append(tokens[i:i + block_size + 1])
        self.examples = torch.tensor(self.examples, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i][:-1], self.examples[i][1:]  # Изменено для соответствия размерностей

# Токенизация
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.reverse_vocab[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        return [self.vocab[word] for word in text.split()]

    def decode(self, tokens):
        return ' '.join([self.reverse_vocab[token] for token in tokens])

# Позиционное кодирование
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Блок трансформера
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# GPT модель
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model*4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc(output)
        return output

# Обучение модели
def train_model(model, dataloader, optimizer, criterion, epochs, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, target in dataloader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(batch.T)
            output = output.permute(1, 2, 0)  # Изменение размерности для совместимости с criterion
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Генерация текста
def generate_text(model, tokenizer, start_text, max_length, device):
    model.eval()
    tokens = tokenizer.encode(start_text)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
    generated = input_tensor

    for _ in range(max_length):
        with torch.no_grad():
            output = model(generated)
            next_token = output[-1, 0, :].argmax().unsqueeze(0).unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=0)
    
    return tokenizer.decode(generated.squeeze().cpu().numpy())

# Пример использования
if __name__ == "__main__":
    # Пример текстов для обучения
    texts = [
        "Привет как дела",
        "Какой сегодня день",
        "Это пример текста для обучения модели",
        "GPT модель на PyTorch",
        "Сегодня отличный день для программирования",
        "Мне нравится изучать машинное обучение",
        "Нейронные сети являются мощным инструментом",
        "Программирование на Python доставляет удовольствие",
        "Машинное обучение изменяет мир",
        "Обработка естественного языка очень интересна",
        "Я люблю изучать новые технологии",
        "Эта модель генерирует текст",
        "Мы можем обучать модели на больших данных",
        "Глубокое обучение открывает новые возможности",
        "Эти примеры помогут в обучении",
        "Модель GPT основана на трансформерах",
        "Нейронные сети обучаются на данных",
        "Генерация текста с помощью модели GPT",
        "Эта реализация использует библиотеку PyTorch",
        "Оптимизация гиперпараметров важна для моделей",
        "Сохранение и загрузка моделей",
        "Обучение и тестирование моделей машинного обучения",
        "Эта модель может предсказывать следующие слова в предложении",
        "Трансформеры широко используются в NLP задачах",
        "Обучение модели требует много вычислительных ресурсов",
        "Эта модель использует многоголовое внимание",
        "Токенизация текста важна для NLP задач",
        "Мы используем слои нормализации в модели",
        "Функция потерь помогает обучать модель",
        "Генерация осмысленного текста с помощью нейросетей",
        "Современные технологии играют важную роль в развитии экономики, поскольку они способствуют увеличению производительности труда, оптимизации процессов и повышению конкурентоспособности предприятий",
        "Образование является ключевым элементом социальной мобильности, поскольку квалифицированный трудовой кадр способствует инновационному развитию общества и повышению уровня жизни граждан.",
        "Защита окружающей среды необходима для сохранения биоразнообразия и поддержания экологического баланса, что в свою очередь обеспечивает устойчивое развитие человечества."
    ]

    # Параметры модели
    d_model = 256  # Уменьшение размера модели для более стабильного обучения
    num_heads = 4  # Соответственно уменьшение числа голов
    num_layers = 4  # Уменьшение числа слоев для ускорения обучения
    block_size = 5
    batch_size = 4
    epochs = 100  # Увеличение числа эпох для более долгого обучения
    learning_rate = 0.0003  # Снижение скорости обучения для более стабильного обучения

    # Инициализация токенизатора и построение словаря
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    # Создание датасета и загрузчика данных
    dataset = TextDataset(texts, tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели, оптимизатора и функции потерь
    model = GPT(vocab_size=tokenizer.vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, optimizer, criterion, epochs, device)

    # Генерация текста
    start_text = "Привет"
    max_length = 10
    generated_text = generate_text(model, tokenizer, start_text, max_length, device)
    print(f"Generated text: {generated_text}")
