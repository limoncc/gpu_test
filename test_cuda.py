import torch
import time

# 配置设备和矩阵大小
device = torch.device("cuda")  # M1/M2 GPU
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
torch.cuda.synchronize()
start_time = time.time()
for _ in range(repeats):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
elapsed_time = time.time() - start_time


# 计算 TFLOPS
total_ops = 2 * n ** 3 * repeats  # 矩阵乘法运算次数：2 * n^3（乘加各一次）
tflops = (total_ops / elapsed_time) / 1e12  # 转换为 TFLOPS

print(f"Time: {elapsed_time:.3f}s, TFLOPS: {tflops:.2f}")

# 3090 bfloat16 Time: 2.368s, TFLOPS: 74.29

# ==================================================
# 各显卡算力测试8192方阵计算矩阵乘法
# ==================================================
# 英伟达A800  float16 Time:  0.782s, TFLOPS: 224.96
# 昇腾910B    float16 Time:  0.948s, TFLOPS: 185.52
# 英伟达4090  float16 Time:  1.116s, TFLOPS: 157.70
# 英伟达H20   float16 Time:  1.235s, TFLOPS: 142.40
# 英伟达A6000 float16 Time:  1.502s, TFLOPS: 117.15
# 英伟达L20   float16 Time:  1.514s, TFLOPS: 116.19
# 摩尔TS4000  float16 Time:  2.222s, TFLOPS:  79.18
# 英伟达3090  float16 Time:  2.384s, TFLOPS:  73.80
# 苹果M1pro   float16 Time: 59.745s, TFLOPS:   2.94
# ==================================================





