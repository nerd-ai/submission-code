import os
import json
import numpy as np
from scipy.special import kl_div

def compute_kl_divergence(prob_map1, prob_map2):
    """
    计算两个概率图的 KL Divergence
    """

    # 检查是否存在全零的概率图
    if np.sum(prob_map1) == 0 or np.sum(prob_map2) == 0:
        print("Warning: One of the probability maps is all zeros.")
        return float(10)  # 如果一个概率图全为零，返回无穷大
    # 确保概率图是归一化的
    prob_map1 = prob_map1 / np.sum(prob_map1)
    prob_map2 = prob_map2 / np.sum(prob_map2)

    # 防止出现零概率导致 log(0) 的情况，使用一个小常数 epsilon 替代零值
    epsilon = 1e-8
    prob_map1 = np.clip(prob_map1, epsilon, 1)
    prob_map2 = np.clip(prob_map2, epsilon, 1)

    # 计算 KL Divergence
    kl_divergence = np.sum(kl_div(prob_map1, prob_map2))
    return kl_divergence

def calculate_kl_for_matching_files(npy_folder1, npy_folder2, output_json_file):
    # 获取两个文件夹中所有 .npy 文件的文件名
    npy_files1 = [f for f in os.listdir(npy_folder1) if f.endswith('.npy')]
    npy_files2 = [f for f in os.listdir(npy_folder2) if f.endswith('.npy')]

    # 只考虑两个文件夹中名称相同的文件
    matching_files = set(npy_files1) & set(npy_files2)

    if not matching_files:
        print("No matching .npy files found.")
        return

    per_mask_metrics = {}
    kl_list = []
    count = 0

    # 遍历每一对匹配的 .npy 文件，计算 KL Divergence
    for file_name in matching_files:
        npy_file1 = os.path.join(npy_folder1, file_name)
        npy_file2 = os.path.join(npy_folder2, file_name)

        # 加载 .npy 数据
        data1 = np.load(npy_file1)
        data2 = np.load(npy_file2)

        # 计算 KL Divergence
        kl_value = compute_kl_divergence(data1, data2)
        per_mask_metrics[file_name] = {"KL": float(kl_value)}

        # 存储KL值用于计算均值和标准差
        kl_list.append(kl_value)
        count += 1

    # 计算平均 KL Divergence 和标准差
    mean_kl = np.mean(kl_list) if count > 0 else 0
    std_kl = np.std(kl_list) if count > 0 else 0

    # 构建结果字典
    result = {
        "overall_metrics": {
            "mean_KL": float(mean_kl),
            "std_KL": float(std_kl)
        },
        "per_mask_metrics": per_mask_metrics
    }

    # 将结果保存到 JSON 文件
    with open(output_json_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"KL Divergence calculations saved to {output_json_file}")

# 示例用法
npy_folder1 = "/path/to/low_temperature_confidence_map"  # 第一个 .npy 文件夹
npy_folder2 = "/path/to/high_temperature_confidence_map"  # 第二个 .npy 文件夹
output_json_file = "kl_results.json"  # 输出的 JSON 文件路径

calculate_kl_for_matching_files(npy_folder1, npy_folder2, output_json_file)