def compute_mean_and_variance(a, b):
    values = [a, b]
    mean = sum(values) / 2
    variance = sum((x - mean) ** 2 for x in values) / 2
    return mean, variance

# 示例：
a = 0.503274490581748
b = 0.4705238131521702
mean, variance = compute_mean_and_variance(a, b)

print(f"Mean: {mean}")
print(f"Variance: {variance}")