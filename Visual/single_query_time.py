import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [0.01, 0.1, 1]

plt.plot(x, y, marker='o')
plt.yscale('log')  # 设置纵坐标为对数坐标
plt.ylabel('Query Time (s)')
plt.grid(True, which='both', ls='--')
plt.show()
