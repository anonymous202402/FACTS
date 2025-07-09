import torch
# NumPy might still be needed for type hints or specific checks if desired,
# but core computation will use torch.
import numpy as np
from typing import List, Tuple, Union

# Constants for trend categories (can be kept as Python ints or made tensors if needed)
TREND_UP = 0
TREND_DOWN = 1
TREND_FLAT = 2

# Small epsilon for variance check (constant detection)
VAR_EPSILON = 1e-9 # Adjust if needed based on expected data scale and precision

def analyze_time_series_trend_gpu(
    time_series_data: torch.Tensor,
    threshold: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分析单变量、多变量或批处理的多变量时间序列数据（或 patch）的趋势 - GPU 版本。

    使用 PyTorch 在 GPU 上计算线性回归斜率。

    - 1D input (length,): 分析单个序列。
    - 2D input (length, num_var): 对每个变量（列）独立分析。
    - 3D input (batch_size, length, num_var): 对每个批次中的每个变量独立分析。

    Args:
        time_series_data (torch.Tensor): 输入的时间序列数据 PyTorch Tensor (应位于目标 GPU 设备)。
                                         期望形状为 1D, 2D 或 3D。
        threshold (float, optional): 判断平稳趋势的斜率阈值。 默认为 1e-3。


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 一个包含两个 Tensor 的元组: (trends, slopes)
            - trends: 包含趋势类别的 Tensor (dtype=torch.long)。形状与输入匹配（去除 length 维度）。
            - slopes: 包含计算出的斜率的 Tensor (dtype=torch.float32 或与输入匹配)。形状与输入匹配（去除 length 维度）。
            - 对于 1D 输入 -> (scalar_tensor, scalar_tensor)
            - 对于 2D 输入 -> (Tensor[num_var], Tensor[num_var])
            - 对于 3D 输入 -> (Tensor[batch_size, num_var], Tensor[batch_size, num_var])
    """
    if not isinstance(time_series_data, torch.Tensor):
        raise TypeError("Input must be a PyTorch Tensor for GPU computation.")
    if not time_series_data.is_cuda:
         # Warn or raise error if data isn't on GPU? For now, assume it is.
         # Or automatically move: time_series_data = time_series_data.to(torch.device("cuda"))
         pass # Assume caller places data on GPU

    device = time_series_data.device
    input_dtype = time_series_data.dtype
    # Use float32 for calculations for stability, unless input is float64
    calc_dtype = torch.float32 if input_dtype != torch.float64 else torch.float64
    data = time_series_data.to(calc_dtype) # Ensure float for calculations

    dim = data.ndim
    time_dim: int

    if dim == 1:
        # Reshape to (L, 1) to reuse 2D logic
        data = data.unsqueeze(-1)
        time_dim = 0
        n_timesteps, n_features = data.shape
        is_batched = False
    elif dim == 2:
        # Shape (L, V)
        time_dim = 0
        n_timesteps, n_features = data.shape
        is_batched = False
    elif dim == 3:
        # Shape (B, L, V)
        time_dim = 1 # Time is the middle dimension
        batch_size, n_timesteps, n_features = data.shape
        is_batched = True
    else:
        raise ValueError("Input Tensor must be 1D, 2D, or 3D.")

    # --- Check minimum length ---
    if n_timesteps < 2:
        if dim == 1:
            return (torch.tensor(TREND_FLAT, device=device, dtype=torch.long),
                    torch.tensor(0.0, device=device, dtype=calc_dtype))
        elif dim == 2:
            return (torch.full((n_features,), TREND_FLAT, device=device, dtype=torch.long),
                    torch.zeros(n_features, device=device, dtype=calc_dtype))
        else: # dim == 3
             return (torch.full((batch_size, n_features), TREND_FLAT, device=device, dtype=torch.long),
                     torch.zeros(batch_size, n_features, device=device, dtype=calc_dtype))

    # --- Prepare x axis tensor ---
    # Shape: (L,) -> unsqueeze to broadcast correctly
    x = torch.arange(n_timesteps, device=device, dtype=calc_dtype)
    if dim == 1: # Originally 1D, now (L, 1)
        x_unsqueezed = x.unsqueeze(1) # Shape (L, 1)
    elif dim == 2: # Shape (L, V)
        x_unsqueezed = x.unsqueeze(1) # Shape (L, 1) - broadcasts over V
    else: # dim == 3, Shape (B, L, V)
        x_unsqueezed = x.unsqueeze(0).unsqueeze(-1) # Shape (1, L, 1) - broadcasts over B and V

    # --- Calculate sums for slope formula (vectorized) ---
    N = n_timesteps
    x_sum = torch.sum(x)  # Scalar
    x_sq_sum = torch.sum(x * x) # Scalar

    # Calculate variance along time dimension to detect constant series
    # Keepdim=True simplifies masking later
    y_var = torch.var(data, dim=time_dim, unbiased=False, keepdim=True)
    is_constant = y_var < VAR_EPSILON

    # Perform sums along the time dimension
    y_sum = torch.sum(data, dim=time_dim, keepdim=True)
    xy_sum = torch.sum(x_unsqueezed * data, dim=time_dim, keepdim=True)

    # Calculate slope components
    denominator = N * x_sq_sum - x_sum * x_sum
    numerator = N * xy_sum - x_sum * y_sum

    # --- Calculate slope, handling division by zero / constant case ---
    slopes = torch.zeros_like(numerator) # Initialize slopes as zero
    # Avoid division by zero if denominator is ~0 (should only happen if N<2 or x is constant, N>=2 handled)
    if abs(denominator.item()) > VAR_EPSILON: # Check denominator safely on CPU
         slopes = numerator / denominator

    # Set slope to 0 where variance was near zero (constant y)
    slopes[is_constant] = 0.0

    # Remove the squeezed time dimension
    slopes = slopes.squeeze(time_dim)

    # --- Determine trend categories ---
    threshold_tensor = torch.tensor(threshold, device=device, dtype=calc_dtype)
    trends = torch.full_like(slopes, TREND_FLAT, dtype=torch.long) # Default to flat
    trends[slopes > threshold_tensor] = TREND_UP
    trends[slopes < -threshold_tensor] = TREND_DOWN

    # Squeeze the result if the original input was 1D
    if dim == 1:
       trends = trends.squeeze()
       slopes = slopes.squeeze()

    return trends, slopes


# Patch function remains largely the same, just update type hints
def patch_time_series_gpu(
    time_series_data: torch.Tensor,
    patch_size: int = 10,
    axis: int = 0
) -> List[torch.Tensor]:
    """
    将(多变量)时间序列数据（PyTorch Tensor）沿指定轴分割成固定大小的块（patches）- GPU兼容。

    支持 1D, 2D, 3D 及更高维度输入。保持数据在原始设备上。
    如果时间序列的总长度不能被 patch_size 整除，
    则最后一个 patch 会与之前的 patch 重叠，以保证其长度为 patch_size
    并包含时间序列的最后的数据点。

    Args:
        time_series_data (torch.Tensor): 输入的时间序列 PyTorch Tensor。
        patch_size (int, optional): 每个 patch 沿指定 `axis` 的期望长度。默认为 10。
        axis (int, optional): 指定哪个轴代表时间/序列长度。
                              - 对于 (length,) -> axis=0
                              - 对于 (length, num_var) -> axis=0
                              - 对于 (batch, length, num_var) -> axis=1
                              - 对于 (batch, num_var, length) -> axis=2

    Returns:
        List[torch.Tensor]: 一个包含所有 patches (作为 Tensor) 的列表。
                            每个 patch 与输入数据在同一设备上，并保持其他维度。

    Raises:
        ValueError: 如果 patch_size 小于或等于 0，或者 axis 无效。
        IndexError: 如果指定的 axis 超出数组/张量维度。
    """
    if not isinstance(time_series_data, torch.Tensor):
         raise TypeError("Input must be a PyTorch Tensor.")

    if patch_size <= 0:
        raise ValueError("patch_size 必须是正整数")
    if time_series_data.ndim == 0:
        return []
    # Allow negative axis indexing like NumPy/PyTorch
    if axis < -time_series_data.ndim or axis >= time_series_data.ndim:
        raise ValueError(f"无效的 axis {axis} for input with {time_series_data.ndim} dimensions")
    # Handle negative axis
    if axis < 0:
        axis += time_series_data.ndim

    n = time_series_data.shape[axis] # Length along the specified axis
    patches: List[torch.Tensor] = []

    if n == 0:
        return patches
    if n < patch_size:
        patches.append(time_series_data)
        return patches

    start_index = 0
    slicer = [slice(None)] * time_series_data.ndim

    while start_index + patch_size <= n:
        end_index = start_index + patch_size
        slicer[axis] = slice(start_index, end_index)
        patches.append(time_series_data[tuple(slicer)])
        start_index += patch_size

    if start_index < n:
        last_patch_start = n - patch_size
        slicer[axis] = slice(last_patch_start, n)
        patches.append(time_series_data[tuple(slicer)])

    return patches

def patch_analyze_trends_gpu(
    time_series: torch.Tensor,
    patch_size: int = 96,
    time_axis: int = 1 # Default for (batch, length, num_var)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Patches a time series tensor and analyzes the trend for each patch on the GPU.

    Args:
        time_series (torch.Tensor): Input time series tensor (e.g., shape [B, L, V]).
                                    Should be on the target GPU device.
        patch_size (int, optional): The size of each patch along the time axis. Defaults to 96.
        time_axis (int, optional): The axis representing time/length.
                                   Defaults to 1 for (batch, length, num_var).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - stacked_trends: Tensor containing trend categories for each patch, variable, and batch item.
                              Shape: (num_patches, batch_size, num_var) or similar depending on input dim.
            - stacked_slopes: Tensor containing slopes for each patch, variable, and batch item.
                              Shape: (num_patches, batch_size, num_var) or similar depending on input dim.
    """
    if not isinstance(time_series, torch.Tensor):
        raise TypeError("Input time_series must be a PyTorch Tensor.")

    # 1. Patch the time series - stays on the original device
    patches: List[torch.Tensor] = patch_time_series_gpu(
        time_series,
        patch_size=patch_size,
        axis=time_axis
    )

    if not patches: # Handle empty input case
        # Determine expected output shape based on input dims excluding time_axis
        output_shape = list(time_series.shape)
        del output_shape[time_axis] # Remove time dimension
        output_shape.insert(0, 0) # Add zero for patch dimension
        empty_trends = torch.empty(output_shape, dtype=torch.long, device=time_series.device)
        empty_slopes = torch.empty(output_shape, dtype=time_series.dtype, device=time_series.device)
        return empty_trends, empty_slopes


    all_patches_trends = []
    all_patches_slopes = []

    # 2. Analyze trend for each patch - calculation stays on GPU
    for patch in patches:
        # analyze_time_series_trend_gpu expects Tensor input and returns Tensor outputs
        # Input patch shape e.g., (B, P, V)
        # Output trends/slopes shapes e.g., (B, V)
        trends_tensor, slopes_tensor = analyze_time_series_trend_gpu(patch) # Removed index [0]

        all_patches_trends.append(trends_tensor)
        all_patches_slopes.append(slopes_tensor)

    # 3. Stack results into single tensors - stays on GPU
    # Stacking along a new dimension (dim=0) -> Shape (num_patches, B, V)
    stacked_trends = torch.stack(all_patches_trends, dim=0)
    stacked_slopes = torch.stack(all_patches_slopes, dim=0)

    return stacked_trends


# --- Modified and Renamed Mispatch Calculation -> Accuracy Calculation ---

def calculate_accuracy_gpu(true_labels: torch.Tensor, pred_labels: torch.Tensor) -> torch.Tensor:
    """
    计算两组标签 Tensor 之间的准确率 (Accuracy) - GPU 版本。

    Args:
        true_labels (torch.Tensor): 包含真实标签的 Tensor (整数类型)。
                                    应位于目标 GPU 设备。
        pred_labels (torch.Tensor): 包含预测标签的 Tensor (整数类型)。
                                    应与 true_labels 形状相同，位于同一设备。

    Returns:
        torch.Tensor: 一个标量 Tensor，表示准确率 (0.0 到 1.0 之间)，位于同一设备。

    Raises:
        ValueError: 如果输入 Tensors 的形状不匹配。
        TypeError: 如果输入不是 PyTorch Tensors。
    """
    if not isinstance(true_labels, torch.Tensor) or not isinstance(pred_labels, torch.Tensor):
        raise TypeError("Inputs must be PyTorch Tensors.")
    if true_labels.shape != pred_labels.shape:
        raise ValueError(f"Input shapes must match: {true_labels.shape} vs {pred_labels.shape}")
    if true_labels.device != pred_labels.device:
         # Warning or error? Let's just proceed assuming user manages devices.
         pass # Consider adding device matching check/warning

    device = true_labels.device
    num_items = true_labels.numel() # Get total number of elements

    if num_items == 0:
        return torch.tensor(1.0, device=device) # Accuracy is 100% for empty inputs? Or NaN? Let's use 1.0

    # Calculate number of matches directly on GPU
    # (true_labels == pred_labels) creates a boolean tensor
    # torch.sum treats True as 1, False as 0
    num_matches = torch.sum(true_labels == pred_labels)

    # Calculate accuracy. Ensure float division.
    accuracy = num_matches.to(torch.float32) / num_items

    return accuracy

# # --- 示例：结合使用 ---
# if torch.cuda.is_available():
#     print("\n--- 示例：结合使用 GPU 函数 ---")
#     device = torch.device("cuda")
#
#     # 沿用之前的示例数据生成
#     batch_s = 4
#     length = 720
#     num_var = 3
#     patch_s = 96
#     input_batch_data = torch.zeros((batch_s, length, num_var), device=device)
#     # (省略数据填充代码 - 假设 input_batch_data 已在 GPU 上填充好)
#     # ... fill input_batch_data on GPU ...
#     # Example fill:
#     input_batch_data = torch.randint(0, 3, (batch_s, length, num_var), device=device)
#
#
#     # 1. 使用 patch_analyze_trends_gpu 获取趋势标签和斜率
#     # Input: (B, L, V) -> Outputs: (P, B, V)
#     stacked_trends, stacked_slopes = patch_analyze_trends_gpu(
#         input_batch_data,
#         patch_size=patch_s,
#         time_axis=1
#     )
#     print(f"Stacked Trends shape: {stacked_trends.shape}") # e.g., (8, 4, 3)
#     print(f"Stacked Slopes shape: {stacked_slopes.shape}") # e.g., (8, 4, 3)
#
#     # 2. 假设我们有一些“真实”的趋势标签进行比较
#     # 这些标签也应该是 Tensor 并在 GPU 上，形状需与 stacked_trends 匹配
#     # 例如，创建一个随机的“真实”标签 Tensor
#     true_trends = torch.randint(0, 3, stacked_trends.shape, device=device, dtype=torch.long)
#
#     # 3. 使用 calculate_accuracy_gpu 计算准确率
#     accuracy_on_gpu = calculate_accuracy_gpu(true_trends, stacked_trends)
#
#     print(f"\nCalculated Accuracy on GPU: {accuracy_on_gpu.item():.4f}") # Use .item() to get Python float for printing
#
# else:
#     print("\nCUDA 不可用，跳过 GPU 结合示例。")