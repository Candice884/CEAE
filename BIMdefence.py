# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:35:24 2025

@author: 17471
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman, SimSun'  # 设置字体族，中文为SimSun，英文为Times New Roman
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams.update({'font.size': 16})  # 设置字体大小
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import LabelEncoder


class RadioML2016Dataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.X = []
        self.Y = []
        self.snr_list = []
        mods = sorted(list(set([k[0] for k in data.keys()])))
        snrs = sorted(list(set([k[1] for k in data.keys()])))

        for mod in mods:
            for snr in snrs:
                if (mod, snr) not in data:
                    continue
                samples = data[(mod, snr)]

                real_part = np.real(samples).astype(np.float32)
                imag_part = np.imag(samples).astype(np.float32)
                merged = np.stack([real_part, imag_part], axis=1)

                self.X.extend(merged)
                self.Y.extend([mod] * samples.shape[0])
                self.snr_list.extend([snr] * samples.shape[0])

        self.X = torch.from_numpy(np.array(self.X))
        self.X = self.X.permute(0, 2, 3, 1)

        le = LabelEncoder()
        self.Y = torch.from_numpy(le.fit_transform(self.Y)).long()
        self.classes = le.classes_
        self.snr_list = torch.tensor(self.snr_list)

        self.mean = self.X.mean(dim=(0, 2, 3), keepdim=True)
        self.std = self.X.std(dim=(0, 2, 3), keepdim=True)
        self.X = (self.X - self.mean) / (self.std + 1e-6)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.snr_list[idx]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1),
                               padding=(1, 0), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1),
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.ca(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        residual = x
        x = self.res_block(x) + residual
        x = self.decoder(x)
        return x * 3.0


def bim_attack(model, data, target, epsilon=0.15, num_iter=40, alpha=0.015):
    perturbed_data = data.clone().detach()
    perturbed_data.requires_grad = True
    
    for _ in range(num_iter):
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_data.grad.data
        sign_data_grad = data_grad.sign()
        
        # 添加扰动
        perturbed_data = perturbed_data + alpha * sign_data_grad
        
        # 计算相对于原始输入的扰动
        perturbation = torch.clamp(perturbed_data - data, -epsilon, epsilon)
        perturbed_data = torch.clamp(data + perturbation, -3, 3).detach()
        perturbed_data.requires_grad = True
    
    return perturbed_data.detach()


def train_dae(dae, cnn, train_loader, val_loader, device):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(dae.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_loss = float('inf')
    early_stop_counter = 0
    epochs = 100

    for epoch in range(epochs):
        dae.train()
        train_loss = 0
        for adv, clean, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            adv, clean = adv.to(device), clean.to(device)

            optimizer.zero_grad()
            recovered = dae(adv)

            loss_mse = mse_loss(recovered, clean)

            with torch.no_grad():
                cnn.eval()
                orig_logits = cnn(clean)
            rec_logits = cnn(recovered)
            loss_ce = ce_loss(rec_logits, orig_logits.argmax(1))

            total_loss = loss_mse + 0.5 * loss_ce
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * adv.size(0)

        dae.eval()
        val_loss = 0
        with torch.no_grad():
            for adv, clean, _, _ in val_loader:
                adv, clean = adv.to(device), clean.to(device)
                recovered = dae(adv)
                val_loss += mse_loss(recovered, clean).item() * adv.size(0)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss * 0.999:
            best_loss = val_loss
            torch.save(dae.state_dict(), "best_dae_BIM.pth") 
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 15:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RadioML2016Dataset("RML2016.10a1.pkl")
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 加载预训练的CNN模型
    cnn = CNN(num_classes=len(dataset.classes)).to(device)
    cnn.load_state_dict(torch.load("CNNbest_model2.pth"))
    cnn.eval()

    # 创建对抗样本数据集
    def create_dae_dataset(subset):
        loader = DataLoader(subset, batch_size=1024, shuffle=False)
        all_adv, all_clean, all_labels, all_snrs = [], [], [], []
    
        for inputs, labels, snrs in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            adv = bim_attack(cnn, inputs, labels, epsilon=0.15, num_iter=40, alpha=0.015)
            all_adv.append(adv.cpu())
            all_clean.append(inputs.cpu())
            all_labels.append(labels.cpu())
            all_snrs.append(snrs)
    
        return TensorDataset(
            torch.cat(all_adv),
            torch.cat(all_clean),
            torch.cat(all_labels),
            torch.cat(all_snrs)
        )
    
    dae_train_set = create_dae_dataset(train_set)
    dae_val_set = create_dae_dataset(val_set)

    train_loader = DataLoader(dae_train_set, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dae_val_set, batch_size=512, shuffle=False)

    # 训练或加载DAE模型
    dae = DAE().to(device)
    model_path = "best_dae_BIM.pth" 

    if os.path.exists(model_path):
        try:
            dae.load_state_dict(torch.load(model_path))
            print("Loaded existing DAE model")
        except:
            print("Existing model incompatible, retraining...")
            os.remove(model_path)
            train_dae(dae, cnn, train_loader, val_loader, device)
            dae.load_state_dict(torch.load(model_path))
    else:
        print("\n--- Training New DAE ---")
        train_dae(dae, cnn, train_loader, val_loader, device)
        dae.load_state_dict(torch.load(model_path))

    # 评估函数
    def evaluate():
        results = {'normal': [], 'adv': [], 'dae': []}
        true_labels = []
        true_snrs = []

        # 正常样本
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
        for inputs, labels, snrs in test_loader:
            inputs = inputs.to(device)
            preds = cnn(inputs).argmax(1).cpu().numpy()
            results['normal'].extend(preds)
            true_labels.extend(labels.numpy())
            true_snrs.extend(snrs.numpy())

        # 对抗样本
        dae_test_set = create_dae_dataset(test_set)
        adv_loader = DataLoader(dae_test_set, batch_size=1024, shuffle=False)
        for adv, _, labels, _ in adv_loader:
            adv = adv.to(device)
            preds = cnn(adv).argmax(1).cpu().numpy()
            results['adv'].extend(preds)

        # DAE修复
        for adv, _, labels, _ in adv_loader:
            adv = adv.to(device)
            with torch.no_grad():
                recovered = dae(adv)
                recovered = torch.clamp(recovered, -3.0, 3.0)
            preds = cnn(recovered).argmax(1).cpu().numpy()
            results['dae'].extend(preds)

        return {k: np.array(v) for k, v in results.items()}, np.array(true_labels), np.array(true_snrs)

    # 评估并获取结果
    results, true_labels, true_snrs = evaluate()
    
    # 计算每个SNR下的平均准确率
    snr_values = sorted(np.unique(true_snrs))
    avg_acc = {'normal': [], 'adv': [], 'dae': []}
    
    for snr in snr_values:
        mask = true_snrs == snr
        if np.sum(mask) == 0:
            continue
            
        # 计算该SNR下的平均准确率
        acc_normal = (results['normal'][mask] == true_labels[mask]).mean()
        acc_adv = (results['adv'][mask] == true_labels[mask]).mean()
        acc_dae = (results['dae'][mask] == true_labels[mask]).mean()
        
        avg_acc['normal'].append(acc_normal)
        avg_acc['adv'].append(acc_adv)
        avg_acc['dae'].append(acc_dae)
    
    # 绘制所有调制类型的平均识别准确率对比图像
    plt.figure(figsize=(12, 8))
    plt.plot(snr_values, avg_acc['normal'], 'b-o', label='No Attack', linewidth=2, markersize=6)
    plt.plot(snr_values, avg_acc['adv'], 'r--s', label='BIM Attack', linewidth=2, markersize=6)
    plt.plot(snr_values, avg_acc['dae'], 'g-.^', label='CEAE Defense', linewidth=2, markersize=6)
    
    plt.title('BIM Attacks CNN Classifier (RML2016.10a dataset)', fontsize=20)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    plt.ylim(-0.05, 1.05)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    # 计算SNR>0时的平均准确率
    mask_positive = true_snrs > 0
    if np.sum(mask_positive) > 0:
        acc_normal_positive = (results['normal'][mask_positive] == true_labels[mask_positive]).mean()
        acc_adv_positive = (results['adv'][mask_positive] == true_labels[mask_positive]).mean()
        acc_dae_positive = (results['dae'][mask_positive] == true_labels[mask_positive]).mean()
        
        # 计算SNR>0时的攻击成功率和防御成功率
        if acc_normal_positive > 0:
            attack_success_rate = 1 - (acc_adv_positive / acc_normal_positive)
            defense_success_rate = (acc_dae_positive - acc_adv_positive) / (acc_normal_positive - acc_adv_positive)
            
    #         plt.figtext(0.15, 0.15, f'SNR>0时统计:\n正常: {acc_normal_positive:.2%}\n攻击: {acc_adv_positive:.2%}\n防御: {acc_dae_positive:.2%}\n攻击成功率: {attack_success_rate:.2%}\n防御成功率: {defense_success_rate:.2%}', 
    #                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    #                     fontsize=14)
    
    # plt.tight_layout()
    # plt.savefig('FGSM_Defense_Comparison_RML2016.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 打印SNR>0时的统计信息
    print("\nSNR>0时的平均准确率统计:")
    print(f"正常样本平均准确率: {acc_normal_positive:.4f}")
    print(f"攻击样本平均准确率: {acc_adv_positive:.4f}")
    print(f"防御样本平均准确率: {acc_dae_positive:.4f}")
    
    # 计算防御效果提升
    defense_improvement = acc_dae_positive - acc_adv_positive
    print(f"防御效果提升: {defense_improvement:.4f}")
    
    # 计算攻击成功率
    if acc_normal_positive > 0:
        attack_success_rate = 1 - (acc_adv_positive / acc_normal_positive)
        print(f"攻击成功率: {attack_success_rate:.2%}")
    else:
        print("攻击成功率: 无法计算（正常样本准确率为0）")
    
    # 计算防御成功率
    if acc_normal_positive - acc_adv_positive > 0:
        defense_success_rate = (acc_dae_positive - acc_adv_positive) / (acc_normal_positive - acc_adv_positive)
        print(f"防御成功率: {defense_success_rate:.2%}")
    else:
        print("防御成功率: 无法计算（攻击未造成性能下降）")


if __name__ == "__main__":
    main()