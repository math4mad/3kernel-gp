import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 模拟数据生成 ====================

def generate_basketball_data(n_samples=500, noise=0.1, random_state=42):
    """
    生成模拟的篮球回合数据
    
    参数:
        n_samples: 样本数量
        noise: 位置噪声
        random_state: 随机种子
    
    返回:
        X: 张量 (n_samples, 2, 5, 2) - [样本, 队伍(己方/对手), 球员, 坐标(x,y)]
        y: 标签 (n_samples,) - 0=失败, 1=成功
    """
    np.random.seed(random_state)
    X = np.zeros((n_samples, 2, 5, 2))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 进攻成功 vs 失败的不同模式
        
        if np.random.rand() > 0.5:  # 进攻成功（标签1）
            # 己方阵型：挡拆成功，球员分散
            # 控球后卫(0)和得分后卫(1)靠近（挡拆）
            X[i, 0, 0] = [0.3, 0.5]  # 控卫
            X[i, 0, 1] = [0.35, 0.45]  # 得分后卫（靠近控卫）
            # 其他球员拉开空间
            X[i, 0, 2] = [0.1, 0.8]  # 小前锋
            X[i, 0, 3] = [0.7, 0.8]  # 大前锋
            X[i, 0, 4] = [0.5, 0.2]  # 中锋（在篮下）
            
            # 对手防守：被拉开，防守阵型被破坏
            X[i, 1, 0] = [0.4, 0.6]  # 对手控卫
            X[i, 1, 1] = [0.5, 0.5]  # 对手得分后卫
            X[i, 1, 2] = [0.2, 0.7]  # 对手小前锋
            X[i, 1, 3] = [0.6, 0.7]  # 对手大前锋
            X[i, 1, 4] = [0.5, 0.3]  # 对手中锋（协防不到位）
            
            y[i] = 1  # 成功
            
        else:  # 进攻失败（标签0）
            # 己方阵型：球员之间距离过大，缺乏配合
            X[i, 0, 0] = [0.2, 0.3]  # 控卫
            X[i, 0, 1] = [0.6, 0.4]  # 得分后卫（远离控卫）
            X[i, 0, 2] = [0.3, 0.7]  # 小前锋
            X[i, 0, 3] = [0.7, 0.6]  # 大前锋
            X[i, 0, 4] = [0.4, 0.2]  # 中锋（在篮下，但孤立）
            
            # 对手防守：收缩防守，协防紧密
            X[i, 1, 0] = [0.3, 0.4]  # 对手控卫
            X[i, 1, 1] = [0.5, 0.4]  # 对手得分后卫
            X[i, 1, 2] = [0.3, 0.6]  # 对手小前锋
            X[i, 1, 3] = [0.5, 0.6]  # 对手大前锋
            X[i, 1, 4] = [0.4, 0.3]  # 对手中锋（协防到位）
            
            y[i] = 0  # 失败
        
        # 添加噪声
        X[i] += np.random.randn(2, 5, 2) * noise
    
    return X, y

# ==================== 2. 三核特征提取 ====================

def compute_rbf_kernel(x1, x2, sigma=1.0):
    """
    计算两个点之间的RBF核
    """
    diff = x1 - x2
    return np.exp(-np.sum(diff**2) / (2 * sigma**2))

def compute_three_kernel_features(X, sigma=1.0):
    """
    计算三核特征
    
    参数:
        X: 张量 (n_samples, 2, 5, 2)
        sigma: RBF核的长度尺度
    
    返回:
        features: 特征矩阵 (n_samples, 3)
    """
    n_samples = X.shape[0]
    features = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        self_pos = X[i, 0]  # (5, 2) - 己方球员位置
        opp_pos = X[i, 1]   # (5, 2) - 对手球员位置
        
        # K_self: 己方阵型紧凑度
        k_self = 0.0
        for p in range(5):
            for q in range(5):
                k_self += compute_rbf_kernel(self_pos[p], self_pos[q], sigma)
        # 归一化：除以5*5=25，使值在[0,1]之间
        k_self /= 25.0
        
        # K_opponent: 对手阵型紧凑度
        k_opponent = 0.0
        for p in range(5):
            for q in range(5):
                k_opponent += compute_rbf_kernel(opp_pos[p], opp_pos[q], sigma)
        k_opponent /= 25.0
        
        # K_cross: 对抗强度（己方vs对手）
        k_cross = 0.0
        for p in range(5):
            for q in range(5):
                k_cross += compute_rbf_kernel(self_pos[p], opp_pos[q], sigma)
        k_cross /= 25.0
        
        features[i] = [k_self, k_opponent, k_cross]
    
    return features

# ==================== 3. 可视化函数 ====================

def plot_court(ax, self_positions, opp_positions, title="Basketball Court"):
    """
    在篮球场上绘制球员位置
    """
    # 绘制球场（简化版）
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制己方球员（蓝色）
    ax.scatter(self_positions[:, 0], self_positions[:, 1], 
               c='blue', s=100, label='Offense', edgecolors='black', zorder=5)
    # 添加球员编号
    for i, pos in enumerate(self_positions):
        ax.annotate(str(i+1), pos, xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, color='blue')
    
    # 绘制对手球员（红色）
    ax.scatter(opp_positions[:, 0], opp_positions[:, 1], 
               c='red', s=100, label='Defense', marker='s', edgecolors='black', zorder=5)
    for i, pos in enumerate(opp_positions):
        ax.annotate(str(i+1), pos, xytext=(5, 5), 
                   textcoords='offset points', fontsize=10, color='red')
    
    # 绘制中圈
    circle = plt.Circle((0.5, 0.5), 0.1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    ax.set_title(title)
    ax.legend(loc='upper right')

def visualize_samples(X, y, n_samples=4):
    """
    可视化一些样本
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    # 选择成功和失败的样本
    success_indices = np.where(y == 1)[0][:n_samples//2]
    failure_indices = np.where(y == 0)[0][:n_samples//2]
    selected_indices = np.concatenate([success_indices, failure_indices])
    
    for idx, ax in zip(selected_indices, axes):
        title = f"Sample {idx} - {'Success' if y[idx] == 1 else 'Failure'}"
        plot_court(ax, X[idx, 0], X[idx, 1], title)
    
    plt.tight_layout()
    plt.show()

# ==================== 4. 主流程 ====================

def main():
    print("=" * 60)
    print("Three-Kernel Gaussian Process Classifier - Basketball Outcome Prediction")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1] Generating synthetic data...")
    X, y = generate_basketball_data(n_samples=500, noise=0.1)
    print(f"    Data shape: {X.shape}")
    print(f"    Label distribution: success={np.sum(y==1)}, failure={np.sum(y==0)}")
    
    # 可视化一些样本
    print("\n[2] Visualizing samples...")
    visualize_samples(X, y, n_samples=4)
    
    # 2. 提取三核特征
    print("\n[3] Extracting three-kernel features...")
    X_features = compute_three_kernel_features(X, sigma=1.0)
    print(f"    Feature shape: {X_features.shape}")
    print("    Feature statistics:")
    for i, name in enumerate(['K_self', 'K_opponent', 'K_cross']):
        print(f"      {name}: mean={X_features[:,i].mean():.3f}, std={X_features[:,i].std():.3f}")
    
    # 3. 分割数据
    print("\n[4] Splitting the dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training set: {X_train.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")
    
    # 4. 训练高斯过程分类器
    print("\n[5] Training the Gaussian process classifier...")
    
    # 定义核函数：常数核 × RBF核 + 白噪声核
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-4, 1e-1))
    
    gpc = GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=5,
        max_iter_predict=100,
        random_state=42
    )
    
    gpc.fit(X_train, y_train)
    
    print(f"    Optimized kernel: {gpc.kernel_}")
    
    # 5. 预测与评估
    print("\n[6] Evaluating the model...")
    
    # 训练集评估
    y_train_pred = gpc.predict(X_train)
    y_train_prob = gpc.predict_proba(X_train)[:, 1]
    
    # 测试集评估
    y_test_pred = gpc.predict(X_test)
    y_test_prob = gpc.predict_proba(X_test)[:, 1]
    
    print("\n    Training results:")
    print(f"      Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"      F1 score: {f1_score(y_train, y_train_pred):.3f}")
    print(f"      AUC:      {roc_auc_score(y_train, y_train_prob):.3f}")
    
    print("\n    Test results:")
    print(f"      Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"      F1 score: {f1_score(y_test, y_test_pred):.3f}")
    print(f"      AUC:      {roc_auc_score(y_test, y_test_prob):.3f}")
    
    # 详细分类报告
    print("\n    Classification report:")
    print(classification_report(y_test, y_test_pred, target_names=['Failure', 'Success']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n    Confusion matrix:")
    print(f"      [[{cm[0,0]:3d}  {cm[0,1]:3d}]")
    print(f"       [{cm[1,0]:3d}  {cm[1,1]:3d}]]")
    
    # 6. 交叉验证
    print("\n[7] Five-fold cross-validation...")
    cv_scores = cross_val_score(gpc, X_features, y, cv=5, scoring='accuracy')
    print(f"    Cross-validation accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    
    # 7. 特征重要性分析（通过单核性能）
    print("\n[8] Single-kernel performance analysis...")
    kernel_names = ['K_self', 'K_opponent', 'K_cross']
    single_kernel_scores = []
    
    for i in range(3):
        X_single = X_features[:, i].reshape(-1, 1)
        kernel_single = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        gpc_single = GaussianProcessClassifier(kernel=kernel_single, random_state=42)
        scores = cross_val_score(gpc_single, X_single, y, cv=5, scoring='accuracy')
        single_kernel_scores.append(scores.mean())
        print(f"    {kernel_names[i]}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # 8. 可视化决策边界（在3D特征空间中）
    print("\n[9] Generating feature distribution plots...")
    fig = plt.figure(figsize=(15, 5))
    
    # 2D散点图
    for idx, name in enumerate(kernel_names):
        ax = fig.add_subplot(1, 3, idx+1)
        for label in [0, 1]:
            mask = y == label
            ax.scatter(X_features[mask, idx], np.zeros(np.sum(mask)), 
                      c=['red' if label==0 else 'blue'][0], 
                      alpha=0.5, label='Failure' if label == 0 else 'Success')
        ax.set_xlabel(name)
        ax.set_ylabel('Kernel density')
        ax.set_title(f'{name} Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 9. 预测概率分布
    print("\n[10] Prediction probability distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 训练集概率
    axes[0].hist(y_train_prob[y_train==0], bins=20, alpha=0.5, label='Failure', color='red')
    axes[0].hist(y_train_prob[y_train==1], bins=20, alpha=0.5, label='Success', color='blue')
    axes[0].set_xlabel('Predicted probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Training Prediction Probabilities')
    axes[0].legend()
    
    # 测试集概率
    axes[1].hist(y_test_prob[y_test==0], bins=20, alpha=0.5, label='Failure', color='red')
    axes[1].hist(y_test_prob[y_test==1], bins=20, alpha=0.5, label='Success', color='blue')
    axes[1].set_xlabel('Predicted probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Test Prediction Probabilities')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()