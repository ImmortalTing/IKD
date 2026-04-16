import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# --- 全局样式配置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11

# 颜色定义
COLOR_X = '#2c7fb8'       # 原始/基准 (蓝色)
COLOR_X_ADV = '#feb24c'   # 对抗样本 (橙色)
COLOR_ERROR = '#e31a1c'   # 误差/符号翻转 (红色)
COLOR_SDS = '#7fcdbb'     # 共享判别子空间 (SDS)
COLOR_BOUNDARY = '#333333'

def add_bottom_label(ax, label):
    """为子图底部中央添加标签 (a, b, c, d)"""
    ax.text(0.5, -0.3, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='center')

def draw_probability_weighting(ax):
    """1. 概率感知权重 (KL vs MSE)"""
    indices = np.arange(6)
    p = np.array([0.8, 0.1, 0.04, 0.03, 0.02, 0.01]) # 原始分布
    weights = 1.0 / (p + 1e-6) # D^-1 权重
    
    ax.bar(indices, p, color=COLOR_X, alpha=0.6, label='$P(x)$')
    ax.set_ylabel('Probability', color=COLOR_X)
    
    ax2 = ax.twinx()
    ax2.plot(indices, weights, color=COLOR_ERROR, marker='o', markersize=4, label='$D^{-1}$')
    ax2.set_ylabel('Weight ($1/P_i$)', color=COLOR_ERROR)
    
    ax.set_title("Probability-Aware\nWeighting ($D^{-1}$)")
    ax.set_xticks(indices)
    ax.set_xticklabels([f'C{i}' for i in indices])
    add_bottom_label(ax, '(a)')

def draw_fim_geometry(ax):
    """2. 费雪信息矩阵与 SDS (几何视角)"""
    theta = np.linspace(0, 2*np.pi, 100)
    
    # KL 的 FIM 椭圆 (长轴对齐 SDS)
    u1 = np.array([1, 0.5]) / np.sqrt(1.25)
    u2 = np.array([-0.5, 1]) / np.sqrt(1.25)
    lambda1, lambda2 = 2.0, 0.4
    
    ell_kl_x = lambda1 * np.cos(theta) * u1[0] + lambda2 * np.sin(theta) * u2[0]
    ell_kl_y = lambda1 * np.cos(theta) * u1[1] + lambda2 * np.sin(theta) * u2[1]
    
    # MSE 的椭圆
    ell_mse_x = 1.0 * np.cos(theta)
    ell_mse_y = 1.0 * np.sin(theta)
    
    ax.plot(ell_kl_x, ell_kl_y, color=COLOR_X, label='KL (FIM)', linewidth=1.5)
    ax.plot(ell_mse_x, ell_mse_y, color=COLOR_BOUNDARY, linestyle='--', label='MSE')
    
    # 画出 SDS 方向
    ax.quiver(0, 0, u1[0]*2, u1[1]*2, color=COLOR_SDS, scale=1, scale_units='xy', label='SDS')
    
    ax.set_title("Geometric View:\nSDS Alignment")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=7)
    add_bottom_label(ax, '(b)')

def draw_gradient_sign_flips(ax):
    """3. 梯度差异与符号翻转 (KL vs CE)"""
    dims = np.arange(30)
    np.random.seed(42)
    grads = np.exp(-dims/8.0) + np.random.normal(0, 0.02, 30)
    grads = np.abs(grads)
    
    epsilon = 0.15
    flipped = grads <= epsilon
    
    # 分别绘制红色点和蓝色点以便添加图例
    stable_idx = ~flipped
    ax.scatter(dims[stable_idx], grads[stable_idx], c=COLOR_X, s=20, 
               edgecolor='white', linewidth=0.5, label='Stable Grad')
    ax.scatter(dims[flipped], grads[flipped], c=COLOR_ERROR, s=20, 
               edgecolor='white', linewidth=0.5, label='Sign Flip')
    
    ax.axhline(y=epsilon, color=COLOR_BOUNDARY, linestyle='--', alpha=0.5)
    ax.fill_between([0, 29], 0, epsilon, color=COLOR_ERROR, alpha=0.1)
    
    ax.set_title("Gradient Stability\n(Sign Flips)")
    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Magnitude $|g_i|$")
    ax.legend(loc='upper right', fontsize=7) # 添加图例
    add_bottom_label(ax, '(c)')

def draw_error_propagation(ax):
    """4. 累积误差传播 (KL vs CE)"""
    t = np.arange(20)
    kt = 5 + np.random.randint(0, 5, 20)
    alpha = 0.01
    upper_bound = 2 * alpha * np.cumsum(np.sqrt(kt))
    actual_error = upper_bound * (0.7 + 0.1 * np.random.rand(20))
    
    ax.plot(t, upper_bound, color=COLOR_BOUNDARY, linestyle='--', label='Bound')
    ax.fill_between(t, 0, actual_error, color=COLOR_ERROR, alpha=0.3)
    ax.plot(t, actual_error, color=COLOR_ERROR, marker='x', markersize=4, label='Actual')
    
    ax.set_title("Error Propagation\nover Steps")
    ax.set_xlabel("Steps $t$")
    ax.set_ylabel("Dist. $\|x_t^{CE} - x_t^{KL}\|$")
    ax.legend(loc='upper left', fontsize=7)
    add_bottom_label(ax, '(d)')

# --- 绘图组装：1行4列 ---
fig = plt.figure(figsize=(15, 4.2)) # 稍微增加高度以容纳底部的标签
gs = gridspec.GridSpec(1, 4, figure=fig)

draw_probability_weighting(fig.add_subplot(gs[0, 0]))
draw_fim_geometry(fig.add_subplot(gs[0, 1]))
draw_gradient_sign_flips(fig.add_subplot(gs[0, 2]))
draw_error_propagation(fig.add_subplot(gs[0, 3]))

# 调整子图间距和边缘，确保底部标签可见
plt.subplots_adjust(left=0.06, right=0.96, top=0.82, bottom=0.22, wspace=0.38)

# fig.suptitle("Theoretical Analysis of Inverse Knowledge Distillation (IKD)", 
#              fontsize=14, fontweight='bold', y=0.96)

plt.savefig("proof1.pdf")
plt.close()