import torch
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # Optimization 1: Use KV cache by storing past_key_values between iterations
    # Optimization 2: Only pass the last generated token after first iteration
    generated_ids = input_ids.clone()
    generated_tokens = []
    past_key_values = None

    for step in range(n_steps):
        if step == 0:
            # First step: pass full context to get initial past_key_values
            outputs = model(input_ids=generated_ids, use_cache=True)
        else:
            # Subsequent steps: only pass the last token, reuse KV cache
            outputs = model(input_ids=generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

    return generated_tokens


def profile(loop_fn, model, input_ids, trace_name: str):
    # Wrap loop_fn with torch.profiler, print the summary table, and export a Chrome trace
    with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            on_trace_ready=lambda p: p.export_chrome_trace(str(RESULTS_DIR / trace_name))
    ) as prof:
        with record_function("generation_loop"):
            loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def generate_optimized(optimized_trace_name: str) -> float:
    # Load model with mixed precision (float16) for faster inference
    model = build_model(torch.float16)
    input_ids = get_input_ids()

    # Profile the optimized loop
    profile(optimized_loop, model, input_ids, optimized_trace_name)

    # Time the generation with MAX_NEW_TOKENS
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    del model
    torch.cuda.empty_cache()

    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v3_optimized_kv_cache_and_fp16.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#
