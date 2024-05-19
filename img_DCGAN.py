import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Определение устройства (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Класс для генератора GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Функция для обучения модели GAN (включая генератор)
def train_gan(generator, dataloader, optimizer, criterion, epochs, device, save_dir):
    generator.to(device)
    generator.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            real_images = batch[0].to(device)
            optimizer.zero_grad()
            z = torch.randn(real_images.size(0), latent_dim).to(device)  # Исправление размера батча
            generated_images = generator(z)
            loss = criterion(generated_images, real_images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Генерация и сохранение изображений после каждой эпохи
        with torch.no_grad():
            z = torch.randn(10, latent_dim).to(device)  # Генерация 10 изображений
            generated_images = generator(z)
            for i, img in enumerate(generated_images):
                os.makedirs(save_dir, exist_ok=True)  # Создание папки для сохранения изображений, если ее нет
                torchvision.utils.save_image(img, f"{save_dir}/generated_{epoch}_{i}.png")

# Функция для генерации изображения на основе пользовательского ввода
def generate_image_from_prompt(generator, prompt, save_path):
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        # Используем промт для генерации шума
        seed = int(prompt) if prompt.isdigit() else 0
        torch.manual_seed(seed)
        z = torch.randn(1, latent_dim).to(device)
        generated_image = generator(z)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(generated_image, save_path)
        print(f"Image saved to {save_path}")

# Гиперпараметры для GAN
latent_dim = 100
img_shape = (1, 28, 28)
batch_size = 64
epochs = 100
learning_rate = 0.00001
save_dir = "generated_images"

# Создание датасета MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Инициализация генератора GAN
generator = Generator(latent_dim, img_shape)

# Оптимизатор и функция потерь для генератора
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
criterion_g = nn.MSELoss()

# Обучение генератора GAN
train_gan(generator, train_loader, optimizer_g, criterion_g, epochs, device, save_dir)

# Получение промта от пользователя и генерация изображения
user_prompt = input("Введите число для генерации изображения: ")
output_path = f"{save_dir}/generated_from_prompt.png"
generate_image_from_prompt(generator, user_prompt, output_path)
