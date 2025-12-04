# 数学概念：离散/连续随机变量、概率分布、期望、方差
# AI对应：生成模型、变分自编码器、概率编程
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributions as dist

# 实践2：概率分布在生成模型中的应用
def probability_distributions_generative_models():
    """
    随机变量和概率分布在生成模型中的应用
    """
    print("=== 概率分布在生成模型中的应用 ===")

    # 1. 常见的概率分布及其性质
    def common_distributions_demo():
        """展示常见概率分布及其在AI中的应用"""
        print("\n1. 常见概率分布及其AI应用:")

        distributions = {
            '高斯分布': {
                'dist': dist.Normal(0, 1),
                'ai_application': 'VAE中的潜在变量、噪声模型',
                'params': '均值μ，标准差σ'
            },
            '伯努利分布': {
                'dist': dist.Bernoulli(0.3),
                'ai_application': '二分类输出、Dropout',
                'params': '成功概率p'
            },
            '均匀分布': {
                'dist': dist.Uniform(0, 1),
                'ai_application': '随机初始化、数据增强',
                'params': '下限a，上限b'
            },
            '指数分布': {
                'dist': dist.Exponential(1.0),
                'ai_application': '生存分析、间隔时间建模',
                'params': '率参数λ'
            },
            'Gamma分布': {
                'dist': dist.Gamma(2.0, 2.0),
                'ai_application': '贝叶斯先验、等待时间',
                'params': '形状α，率β'
            },
            'Beta分布': {
                'dist': dist.Beta(2.0, 5.0),
                'ai_application': '成功率建模、贝叶斯A/B测试',
                'params': '形状参数α, β'
            }
        }

        # 可视化分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for idx, (name, info) in enumerate(distributions.items()):
            ax = axes[idx // 3, idx % 3]#用于在2×3的子图网格中定位特定的子图轴对象。

            # 生成样本并绘制直方图
            samples = info['dist'].sample((10000,))

            if isinstance(info['dist'], dist.Bernoulli):
                # 伯努利分布特殊处理
                unique, counts = np.unique(samples.numpy(), return_counts=True)#使用 np.unique 统计样本中0和1的出现次数
                ax.bar(['0', '1'], counts / counts.sum(), alpha=0.7)#counts / counts.sum() 将频次转换为概率
                ax.set_ylim(0, 1)
            else:
                ax.hist(samples.numpy(), bins=50, density=True, alpha=0.7, edgecolor='black')

            # 计算统计量
            mean = samples.mean().item()#计算均值
            std = samples.std().item()#计算方差
            skew = samples.numpy().mean(axis=0)#计算偏度

            ax.set_title(f'{name}\nμ={mean:.2f}, σ={std:.2f}')
            ax.set_xlabel('值')
            ax.set_ylabel('概率密度')
            ax.grid(True, alpha=0.3)

            # 在图上添加AI应用
            ax.text(0.05, 0.95, f"AI: {info['ai_application']}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    common_distributions_demo()

    # 2. 从高斯混合模型生成数据
    def gaussian_mixture_model_demo():
        """高斯混合模型数据生成"""
        print("\n2. 高斯混合模型生成数据:")

        # 定义三个高斯分布
        n_components = 3
        means = torch.tensor([[-2.0, -2.0], [0.0, 2.0], [2.0, -1.0]])
        covs = torch.tensor([
            [[0.5, 0.2], [0.2, 0.5]],
            [[0.8, -0.3], [-0.3, 0.8]],
            [[0.4, 0.0], [0.0, 0.4]]
        ])#协方差矩阵

        # 混合权重
        weights = torch.tensor([0.3, 0.5, 0.2])

        # 生成数据
        n_samples = 1000 #数据样本数量
        samples = [] #用于存储生成的样本
        labels = [] #用于存储样本的标签

        for i in range(n_samples):
            # 选择组件
            comp = torch.multinomial(weights, 1).item()

            # 从选定的高斯分布生成样本
            mean = means[comp]
            cov = covs[comp]

            '''
            多元高斯分布采样代码解释
            这三行代码实现了从多元高斯分布中生成样本的变换方法：
            代码解析
                L = torch.linalg.cholesky(cov):
                对协方差矩阵 cov 进行 Cholesky 分解
                得到下三角矩阵 L，满足 L @ L.T = cov
            z = torch.randn(2):
                从标准正态分布生成2维随机向量
                每个元素独立服从 N(0,1)
            sample = mean + L @ z:
                通过线性变换将标准正态样本转换为指定均值和协方差的样本
                这基于性质：若 z ~ N(0,I)，则 Lz + μ ~ N(μ, LLᵀ) = N(μ, Σ)
            数学原理
                这是从多元正态分布 N(mean, cov) 采样的标准方法：
                利用 Cholesky 分解将协方差矩阵分解
                通过仿射变换将独立标准正态变量转换为目标分布
                这种方法比直接使用 torch.distributions.MultivariateNormal 更高效，特别是在需要大量采样时。
            '''
            # 使用Cholesky分解生成多元高斯样本
            L = torch.linalg.cholesky(cov)
            z = torch.randn(2)
            sample = mean + L @ z

            samples.append(sample) #存储生成的样本
            labels.append(comp) #存储样本的标签

        samples = torch.stack(samples)
        labels = torch.tensor(labels)

        print(f"生成数据统计:")
        print(f"  样本数量: {n_samples}")
        print(f"  类别分布: {np.bincount(labels.numpy())}")
        print(f"  混合权重: {weights.numpy()}")

        # 可视化
        plt.figure(figsize=(12, 10))

        # 数据散点图
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(samples[:, 0], samples[:, 1], c=labels,
                              cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='组件')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('高斯混合模型生成的数据')
        plt.grid(True, alpha=0.3)

        # 每个组件的概率密度
        plt.subplot(2, 2, 2)
        # 创建网格
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)

        # 计算每个组件的概率密度
        Z_total = torch.zeros(grid_points.shape[0])

        for comp in range(n_components):
            mean = means[comp]
            cov = covs[comp]

            # 多元高斯分布
            mvn = dist.MultivariateNormal(mean, covariance_matrix=cov)
            log_prob = mvn.log_prob(grid_points)
            prob = torch.exp(log_prob) * weights[comp]

            Z_total += prob

            # 绘制单个组件的等高线
            Z = prob.reshape(X.shape).numpy()
            plt.contour(X, Y, Z, levels=3, alpha=0.5, linestyles='--')

        # 绘制混合分布的等高线
        Z_total = Z_total.reshape(X.shape).numpy()
        contour = plt.contour(X, Y, Z_total, levels=10, colors='black', alpha=0.7)
        plt.clabel(contour, inline=True, fontsize=8)

        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('高斯混合模型的概率密度')
        plt.grid(True, alpha=0.3)

        # 边际分布
        plt.subplot(2, 2, 3)
        for comp in range(n_components):
            comp_samples = samples[labels == comp]
            plt.hist(comp_samples[:, 0].numpy(), bins=30, alpha=0.5, density=True,
                     label=f'组件 {comp}')

        plt.xlabel('特征1')
        plt.ylabel('概率密度')
        plt.title('特征1的边际分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        for comp in range(n_components):
            comp_samples = samples[labels == comp]
            plt.hist(comp_samples[:, 1].numpy(), bins=30, alpha=0.5, density=True,
                     label=f'组件 {comp}')

        plt.xlabel('特征2')
        plt.ylabel('概率密度')
        plt.title('特征2的边际分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return samples, labels, weights, means, covs

    gmm_samples, gmm_labels, gmm_weights, gmm_means, gmm_covs = gaussian_mixture_model_demo()

    # 3. 变分自编码器中的概率分布
    def variational_autoencoder_demo():
        """变分自编码器中的概率分布"""
        print("\n3. 变分自编码器中的概率分布:")

        class SimpleVAE(nn.Module):
            def __init__(self, input_dim=2, latent_dim=2, hidden_dim=10):#输入维度，隐空间维度，隐藏层维度
                super().__init__()

                # 编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),#输入层
                    nn.ReLU(),#激活函数
                    nn.Linear(hidden_dim, hidden_dim),#隐藏层
                    nn.ReLU(),#隐藏层激活函数
                )

                # 潜在空间的均值和方差
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

                # 解码器
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),  # 将潜在空间维度映射到隐藏层维度
                    nn.ReLU(),                          # 激活函数，引入非线性
                    nn.Linear(hidden_dim, hidden_dim),   # 隐藏层到隐藏层的映射
                    nn.ReLU(),                          # 激活函数，引入非线性
                    nn.Linear(hidden_dim, input_dim * 2)  # 输出层：映射到输入维度的2倍（均值和方差）
                )


                self.input_dim = input_dim
                self.latent_dim = latent_dim

            def encode(self, x):
                """编码器：将输入数据映射到潜在空间的分布参数"""
                h = self.encoder(x)  # 通过编码器网络提取特征
                mu = self.fc_mu(h)  # 计算潜在变量的均值
                logvar = self.fc_logvar(h)  # 计算潜在变量的对数方差
                return mu, logvar  # 返回均值和对数方差

            def reparameterize(self, mu, logvar):
                """重参数化技巧：实现随机采样同时保持梯度流动"""
                std = torch.exp(0.5 * logvar)  # 从对数方差计算标准差
                eps = torch.randn_like(std)  # 从标准正态分布采样噪声
                return mu + eps * std  # 重参数化：z = μ + σ ⊙ ε

            def decode(self, z):
                """解码器：将潜在变量映射回数据空间的分布参数"""
                h = self.decoder(z)  # 通过解码器网络处理潜在变量
                mu_recon = h[:, :self.input_dim]  # 提取重构数据的均值参数
                logvar_recon = h[:, self.input_dim:]  # 提取重构数据的对数方差参数
                return mu_recon, logvar_recon  # 返回重构分布的参数

            def forward(self, x):
                """前向传播：完整的VAE流程"""
                mu, logvar = self.encode(x)  # 编码输入数据
                z = self.reparameterize(mu, logvar)  # 从编码分布中采样
                mu_recon, logvar_recon = self.decode(z)  # 解码生成重构数据
                return mu_recon, logvar_recon, mu, logvar, z  # 返回所有相关参数

        # 创建VAE模型
        vae = SimpleVAE(input_dim=2, latent_dim=2, hidden_dim=20)

        # 使用GMM（高斯混合模型（Gaussian Mixture Model））数据训练VAE
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.01)

        # 准备数据
        vae_data = gmm_samples.float()#转换为张量
        dataset = torch.utils.data.TensorDataset(vae_data)#创建数据集
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)#创建数据加载器

        # 训练循环
        n_epochs = 100
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0

            for batch in dataloader:
                x = batch[0]

                optimizer.zero_grad()

                # 前向传播
                mu_recon, logvar_recon, mu, logvar, z = vae(x)

                # 计算损失
                # 重构损失（负对数似然）
                recon_loss = 0.5 * torch.sum(
                    logvar_recon + (x - mu_recon).pow(2) / torch.exp(logvar_recon)
                ) / x.size(0)

                # KL散度损失
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

                # 总损失
                loss = recon_loss + kl_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))#添加到列表中，计算每个epoch的平均损失

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {losses[-1]:.4f} "
                      f"(Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f})")

        # 可视化训练过程
        plt.figure(figsize=(12, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(losses, label='总损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('VAE训练损失')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 潜在空间可视化
        plt.subplot(2, 2, 2)
        with torch.no_grad():
            mu, logvar = vae.encode(vae_data)
            z_samples = vae.reparameterize(mu, logvar)

        plt.scatter(z_samples[:, 0].numpy(), z_samples[:, 1].numpy(),
                    c=gmm_labels.numpy(), cmap='viridis', alpha=0.6, s=10)
        plt.xlabel('潜在维度1')
        plt.ylabel('潜在维度2')
        plt.title('编码后的潜在空间')
        plt.colorbar(label='原始类别')
        plt.grid(True, alpha=0.3)

        # 从潜在空间生成新样本
        plt.subplot(2, 2, 3)
        with torch.no_grad():
            # 在潜在空间中采样
            z_new = torch.randn(200, 2) * 1.5  # 从标准正态分布采样
            mu_recon, logvar_recon = vae.decode(z_new)

            # 从重构分布中采样
            std_recon = torch.exp(0.5 * logvar_recon)
            eps = torch.randn_like(std_recon)
            x_recon = mu_recon + eps * std_recon

        plt.scatter(x_recon[:, 0].numpy(), x_recon[:, 1].numpy(), alpha=0.6, s=10)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('从潜在空间生成的新样本')
        plt.grid(True, alpha=0.3)

        # 重构质量
        plt.subplot(2, 2, 4)
        n_show = 100
        indices = torch.randperm(len(vae_data))[:n_show]
        x_original = vae_data[indices]

        with torch.no_grad():
            mu_recon, logvar_recon, _, _, _ = vae(x_original)
            # 使用均值作为重构
            x_reconstructed = mu_recon

        plt.scatter(x_original[:, 0], x_original[:, 1], alpha=0.6, label='原始', s=30)
        plt.scatter(x_reconstructed[:, 0], x_reconstructed[:, 1],
                    alpha=0.6, label='重构', s=30, marker='x')

        # 连接原始和重构点
        for i in range(n_show):
            plt.plot([x_original[i, 0], x_reconstructed[i, 0]],
                     [x_original[i, 1], x_reconstructed[i, 1]],
                     'k-', alpha=0.1, linewidth=0.5)

        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('原始数据与重构数据对比')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nVAE概率解释:")
        print("1. 编码器: q(z|x) = N(z; μ(x), σ²(x)I) - 近似后验分布")
        print("2. 先验: p(z) = N(z; 0, I) - 标准正态分布")
        print("3. 解码器: p(x|z) = N(x; μ(z), σ²(z)I) - 似然函数")
        print("4. 目标: 最大化证据下界 (ELBO)")
        print("5. 重参数化技巧: z = μ + σ ⊙ ε, ε ∼ N(0, I)")

        return vae

    vae_model = variational_autoencoder_demo()

    return vae_model


vae_model = probability_distributions_generative_models()