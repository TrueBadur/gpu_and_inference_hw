import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x.clone()
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    time_ms = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        time_ms.append(start.elapsed_time(end))
    time_ms.sort()
    mid = len(time_ms) // 2
    if len(time_ms) % 2 == 1:
        return time_ms[mid]
    else:
        return (time_ms[mid - 1] + time_ms[mid]) / 2


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    # Each iteration of `acc = acc * x + x` does 2 FLOPs per element
    total_flops = float(num_elements) * float(num_ops) * 2.0

    # For "compiled" variant: the whole loop is fused in one kernel, so each element
    # is read once and written once at the kernel boundary, regardless of num_ops.
    # Traffic = num_elements * (read + write) * bytes_per_element
    #
    # For "eager" variant: each iteration launches separate mul and add kernels:
    #   mul:  read(acc) + read(x) -> write(tmp)
    #   add:  read(tmp) + read(x) -> write(acc)
    # Traffic per iteration ≈ 6 * bytes_per_element per element (4 reads + 2 writes)
    if variant == "compiled":
        # Fused: read + write per element, once
        traffic_bytes = float(num_elements) * float(bytes_per_element) * 2.0
    else:
        # Eager: 6 accesses per iteration per element
        traffic_bytes = float(num_elements) * float(num_ops) * float(bytes_per_element) * 6.0

    if traffic_bytes <= 0.0:
        ai = float("inf")
    else:
        ai = float(total_flops) / float(traffic_bytes)

    seconds = float(ms) / 1000.0
    if seconds <= 0.0:
        achieved_flops = float("inf")
    else:
        achieved_flops = float(total_flops) / float(seconds)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. The top performance GPU reaches when it performs computations on "same" data in
# fast memory. The more ops fused in one kernel, the more GPU can work on the data
# before it is moved to a slower memory. Basically GPU can perform more operations
# in unit of time but there is no more work to do. It's like a truck which can move 100 boxes per race.
# It will take the same amount of time to deliver 1, 50 or 100 boxes.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. 1. The overhead of the fixed additional work made by the pipeline, like
# host-to-device launch latency and grid synchronization dominate the runtime.
# 2. 1024x1024 matrix multiply operation is not filling the whole H100 card capacity
# underutilizing its top performance, while the data used in the 128 ops compiled element-wise kernel (64 x 1024 x 1024) is.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. The bottleneck is now top possible performance of the GPU. The GPU can't work faster.
# Continuing the car example from Q1, now the number of boxes is 150. The truck must have 2 races to deliver them.
# So the amount of time it spends in the way grows.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. With eager execution each operation is run separately, so the data has to be retrieved from and written
# back to slow global memory (HBM) for each operation, which causes a huge traffic and makes GPU spend most of the
# time waiting for data.
