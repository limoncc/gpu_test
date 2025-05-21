import torch
import torch_npu
import time

device = torch.device("npu")
n = 8192*2  # 显存充足时尽量调大
dtype = torch.float16  # 使用半精度加速


# 预分配内存
a = torch.randn(n, n, dtype=dtype, device=device)
b = torch.randn(n, n, dtype=dtype, device=device)
c = torch.empty(n, n, dtype=dtype, device=device)

# 预热（丢弃前 5 次）
for _ in range(10):
    torch.mm(a, b, out=c)

# 精确计时
repeats = 20
opt_mm = torch.compile (torch.mm)
torch.mps.synchronize()
start_time = time.time()
for _ in range(repeats):
    torch.mm(a, b, out=c)
    # opt_mm(a, b, out=c)
torch.mps.synchronize()
elapsed_time = time.time() - start_time

# 计算 TFLOPS（注意：float16 的 FLOPS 计算需根据硬件优化）
total_ops = 2 * n ** 3 * repeats  # 即使使用 float16，运算次数仍按 float32 计数
tflops = (total_ops / elapsed_time) / 1e12

print(f"Time: {elapsed_time:.3f}s, TFLOPS: {tflops:.2f}")

