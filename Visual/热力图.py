import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io

# 创建一个带社区结构感的6x6连接概率矩阵，值在0~1之间
# example_matrix = np.array([
#     [0.0, 0.9, 0.85, 0.2, 0.1, 0.05],
#     [0.9, 0.0, 0.88, 0.25, 0.15, 0.1],
#     [0.85, 0.88, 0.0, 0.22, 0.12, 0.08],
#     [0.2, 0.25, 0.22, 0.0, 0.87, 0.9],
#     [0.1, 0.15, 0.12, 0.87, 0.0, 0.88],
#     [0.05, 0.1, 0.08, 0.9, 0.88, 0.0],
# ])
example_matrix = np.array([
    [0.0, 0.88, 0.20, 0.10, 0.05],
    [0.88, 0.0, 0.25, 0.15, 0.10],
    [0.20, 0.25, 0.0, 0.85, 0.82],
    [0.10, 0.15, 0.85, 0.0, 0.87],
    [0.05, 0.10, 0.82, 0.87, 0.0],
])

# 创建彩色热力图
plt.figure(figsize=(4, 4))
sns.heatmap(example_matrix, cmap='PuBu', square=True, cbar=False,
            xticklabels=False, yticklabels=False, linewidths=0.5, linecolor='black')
plt.axis('off')
plt.tight_layout()

output_file = './visual_results/reli.png'  # 可替换为 'BarChart_F1_Score_InternalTopLegend_NoOverlap.eps'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# plt.show()

