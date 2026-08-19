import time, os, torch, numpy as np
def benchmark(matrix_dim=1024, iterations=500):
    a, b = torch.randn(matrix_dim, matrix_dim), torch.randn(matrix_dim, matrix_dim)
    for _ in range(20): _ = torch.mm(a, b)
    t0 = time.perf_counter()
    for _ in range(iterations): _ = torch.mm(a, b)
    print(f"Elapsed: {time.perf_counter()-t0:.4f}s")
if __name__ == "__main__": benchmark()
