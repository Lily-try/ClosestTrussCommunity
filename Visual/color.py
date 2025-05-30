import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#baokan
# 从 Excel 文件读取数据
file_path = './results/Example_Data.xlsx'  # 请确保文件路径正确
data = pd.read_excel(file_path)

# 从数据中提取列
datasets = data.iloc[:, 0]  # 第一列为数据集 (x轴值)
methods = ["MkECSs", "CTC", "ICS-GNN", "COCLEP", "STABLE", "GNN-Guard", "Ours"]
scores = [data[method] for method in methods]  # 每种方法的分数
stds = [data[f"{method}_std"] for method in methods]  # 每种方法的标准差

# 配置柱状图参数
bar_width = 0.1  # 柱宽
index = np.arange(len(datasets))  # x轴位置

# 设置字体大小
font = {
    'family': 'Arial',  # 可选字体，如 'Times New Roman'
    'size': 14  # 全局字体大小
}
plt.rc('font', **font)

# 创建图形和子图
fig, ax = plt.subplots(figsize=(14, 6))

# 为每种方法绘制柱状图并添加误差线
colors = ['green', 'yellowgreen', 'orange', 'red', 'darkgray', 'purple', 'darkred']
hatches = ['/', '\\', '--', 'xx', '..', 'oo', '++']
for i, (score, std) in enumerate(zip(scores, stds)):
    ax.bar(index + (i - len(scores) / 2) * bar_width, score, bar_width, label=methods[i],
           yerr=std, capsize=5, hatch=hatches[i], color='none', edgecolor=colors[i])

# 添加标签、标题等（调整字体大小）
ax.set_xlabel('Dataset', fontsize=16)  # x轴标签字体
ax.set_ylabel('F1 Score', fontsize=16)  # y轴标签字体
ax.set_title('F1 Score by Method and Dataset Network (with Standard Deviation)', fontsize=18)  # 标题字体
ax.set_xticks(index)
ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=14)  # x轴刻度字体

# 设置图例在图表内部上方，且不覆盖柱状图
ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False)  # 图例在图表内上方留足空间

# 自动调整子图参数以给定的填充
plt.tight_layout(rect=[0, 0, 1, 1])  # 调整rect参数，预留顶部空间

# 保存图片为科研论文常用格式（如 PDF 或 EPS）
output_file = './visual_results/F1_varyPtb.pdf'  # 可替换为 'BarChart_F1_Score_InternalTopLegend_NoOverlap.eps'
plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')  # dpi=300 为高分辨率，bbox_inches 去掉多余空白

# 显示图形
plt.show()
