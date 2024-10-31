import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(32),              # 调整图像尺寸为 32x32
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

batch_size = 128

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


T = 1000  # 扩散步骤数
beta_start = 1e-4
beta_end = 0.02
beta = torch.linspace(beta_start, beta_end, T, device=device)
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

def q_sample(x0, t, epsilon):
    """
    在时间步 t，将噪声 epsilon 添加到原始数据 x0，得到 x_t。
    """
    sqrt_alpha_hat = alpha_hat[t] ** 0.5
    sqrt_one_minus_alpha_hat = (1 - alpha_hat[t]) ** 0.5
    sqrt_alpha_hat = sqrt_alpha_hat.view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat.view(-1, 1, 1, 1)
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * epsilon

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class UNet(nn.Module):
    def __init__(self, img_channels=1, time_emb_dim=256, num_classes=10):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        # 类别嵌入
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # 嵌入映射层（下采样）
        self.emb1 = nn.Linear(time_emb_dim, 64)
        self.emb2 = nn.Linear(time_emb_dim, 128)
        self.emb3 = nn.Linear(time_emb_dim, 256)
        self.emb_mid = nn.Linear(time_emb_dim, 256)

        # 嵌入映射层（上采样）
        self.emb_up3 = nn.Linear(time_emb_dim, 128)
        self.emb_up2 = nn.Linear(time_emb_dim, 64)
        self.emb_up1 = nn.Linear(time_emb_dim, img_channels)

        # 下采样部分
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)

        # 中间层
        self.mid = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
        )

        # 上采样部分
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1),
        )

    def forward(self, x, t, y):
        # 时间嵌入
        t_emb = self.time_mlp(t)
        # 类别嵌入
        class_emb = self.class_emb(y)
        # 融合时间和类别嵌入
        emb = t_emb + class_emb

        # 下采样
        x1 = self.down1(x)
        emb1 = self.emb1(emb).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + emb1
        x2 = self.pool1(x1)

        x2 = self.down2(x2)
        emb2 = self.emb2(emb).unsqueeze(-1).unsqueeze(-1)
        x2 = x2 + emb2
        x3 = self.pool2(x2)

        x3 = self.down3(x3)
        emb3 = self.emb3(emb).unsqueeze(-1).unsqueeze(-1)
        x3 = x3 + emb3
        x4 = self.pool3(x3)

        # 中间层
        x_mid = self.mid(x4)
        emb_mid = self.emb_mid(emb).unsqueeze(-1).unsqueeze(-1)
        x_mid = x_mid + emb_mid

        # 上采样
        x = self.up3(x_mid)

        # 调整尺寸并拼接 x 和 x3
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)
        emb_up3 = self.emb_up3(emb).unsqueeze(-1).unsqueeze(-1)
        x = x + emb_up3

        x = self.up2(x)

        # 调整尺寸并拼接 x 和 x2
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv2(x)
        emb_up2 = self.emb_up2(emb).unsqueeze(-1).unsqueeze(-1)
        x = x + emb_up2

        x = self.up1(x)

        # 调整尺寸并拼接 x 和 x1
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv1(x)
        emb_up1 = self.emb_up1(emb).unsqueeze(-1).unsqueeze(-1)
        x = x + emb_up1

        # 输出预测的噪声
        return x

def train(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for idx, (x0, labels) in enumerate(dataloader):
            x0 = x0.to(device)
            labels = labels.to(device)
            batch_size = x0.size(0)
            t = torch.randint(0, T, (batch_size,), device=device).long()
            epsilon = torch.randn_like(x0).to(device)
            x_t = q_sample(x0, t, epsilon)
            t_norm = t / T  # 归一化时间步
            predicted_epsilon = model(x_t, t_norm.float(), labels)
            loss = nn.MSELoss()(predicted_epsilon, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {idx}/{len(dataloader)} Loss: {loss.item():.4f}")


model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50

train(model, train_loader, optimizer, num_epochs)


@torch.no_grad()
def p_sample(model, x, t, y):
    t_tensor = torch.tensor([t], device=device).long()
    alpha_t = alpha[t_tensor].view(-1, 1, 1, 1)
    alpha_hat_t = alpha_hat[t_tensor].view(-1, 1, 1, 1)
    beta_t = beta[t_tensor].view(-1, 1, 1, 1)

    epsilon_theta = model(x, (t_tensor / T).float(), y)
    coef1 = 1 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1 - alpha_hat_t)

    x_prev = coef1 * (x - coef2 * epsilon_theta)
    if t > 0:
        noise = torch.randn_like(x)
        x_prev = x_prev + torch.sqrt(beta_t) * noise
    return x_prev