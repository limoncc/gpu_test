import torch
import torch_musa
import time

# 配置设备和矩阵大小
device = torch.device("musa")  # M1/M2 GPU
n = 8192*2  # 矩阵大小 (n x n)，根据显存调整
dtype = torch.float16  # 单精度浮点

# 创建随机矩阵并移动到 GPU
a = torch.randn(n, n, dtype=dtype, device=device)
b = torch.randn(n, n, dtype=dtype, device=device)
c = torch.empty ( n , n , device = device , dtype = dtype )

# 预热 GPU（避免初始化开销）
for _ in range(10):
    torch.mm(a, b)


# 正式测试
repeats = 20
# 精确计时
torch.musa.synchronize()
start_time = time.time()
for _ in range(repeats):
    torch.mm(a, b, out=c)
torch.musa.synchronize()
elapsed_time = time.time() - start_time


# 计算 TFLOPS
total_ops = 2 * n ** 3 * repeats  # 矩阵乘法运算次数：2 * n^3（乘加各一次）
tflops = (total_ops / elapsed_time) / 1e12  # 转换为 TFLOPS

print(f"Time: {elapsed_time:.3f}s, TFLOPS: {tflops:.2f}")




