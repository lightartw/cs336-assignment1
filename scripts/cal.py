def calculate_parameters(d, v, d_ff, num_layers):
    # 2*d (RMSNorm) + 4*d^2 (MHA) + 3*d*d_ff (SwiGLU)
    block_params = (2 * d) + (4 * d**2) + (3 * d * d_ff)
    # 2*v*d (Embedding/Logits) + d (RMSNorm) + num_layers * block_params
    total_params = (2 * v * d) + d + (num_layers * block_params)
    
    print("【参数量统计】")
    print(f"单个 Block 参数量: {block_params:,}")
    print(f"模型总参数量:      {total_params:,}")
    return total_params

def calculate_flops(b, n, d, v, d_ff, num_layers):
    # 1. Linear (Logits) 的总 FLOPs: 2 * b * n * d * v
    total_linear_flops = 2 * b * n * d * v
    # 2. MHA 的总 FLOPs: num_layers * (8*b*n*d^2 + 4*b*n^2*d)
    single_layer_mha_flops = (8 * b * n * d**2) + (4 * b * n**2 * d)
    total_mha_flops = num_layers * single_layer_mha_flops
    # 3. SwiGLU 的总 FLOPs: num_layers * (6*b*n*d*d_ff)
    single_layer_swiglu_flops = 6 * b * n * d * d_ff
    total_swiglu_flops = num_layers * single_layer_swiglu_flops
    
    total_flops = total_linear_flops + total_mha_flops + total_swiglu_flops
    
    # 计算百分比
    linear_pct = (total_linear_flops / total_flops) * 100
    mha_pct = (total_mha_flops / total_flops) * 100
    swiglu_pct = (total_swiglu_flops / total_flops) * 100

    print("\n" + "=" * 60)
    print("【计算量统计 (FLOPs - Matmul Only)】")
    print(f"LM 总 Linear FLOPs: {total_linear_flops:,} ({linear_pct:.2f}%)")
    print(f"LM 总 MHA FLOPs:    {total_mha_flops:,} ({mha_pct:.2f}%)")
    print(f"LM 总 SwiGLU FLOPs: {total_swiglu_flops:,} ({swiglu_pct:.2f}%)")
    print(f"模型总计算量:       {total_flops:,} (100.00%)")
    print("=" * 60)
    
    return total_flops

if __name__ == "__main__":
    b = 1
    v = 50257
    n = 1024
    d = 1600
    d_ff = 4 * d
    num_layers = 48

    calculate_parameters(d=d, v=v, d_ff=d_ff, num_layers=num_layers)
    calculate_flops(b=b, n=n, d=d, v=v, d_ff=d_ff, num_layers=num_layers)