import random
import json

def generate_random_image_selection(
    high_iou_txt,
    output_txt,
    total_images=3668, # adjust based on actual total number of images
    ratios=("0.01", "0.05", "0.10", "0.15", "0.20", "0.25","0.50"),
    seeds=7,
    use_all_if_insufficient=True,
):
    """
    从 high_iou_image_ids.txt 中读取 image_id（每行一个），
    针对每个给定比例（如 "0.01", "0.05", "0.10"）计算需要抽取的个数，
    使用 seeds 个不同随机种子生成随机抽样结果，
    并对抽取结果进行排序，
    最后将结果保存到 output_txt 文件中（JSON 格式）。
    """
    # 读取 high_iou_image_ids.txt 文件
    with open(high_iou_txt, 'r') as f:
        lines = f.read().splitlines()
    
    # 转换为整数列表（假设每行仅包含一个数字）
    pool = [int(line.strip()) for line in lines if line.strip()]
    
    # 构造存放结果的字典
    output_data = {}
    
    for ratio_str in ratios:
        ratio = float(ratio_str)
        # 计算需要抽取的数量（取整）
        sample_count = int(total_images * ratio)
        output_data[ratio_str] = {}

        # 当所需数量超过可用数量时，按需求将所有 ID 作为该比例的结果，且只记录 seed=1
        if sample_count > len(pool):
            if use_all_if_insufficient:
                output_data[ratio_str]["1"] = sorted(pool)
                continue  # 跳过该比例的后续随机抽样
            else:
                raise ValueError(
                    f"当前 pool 中的 image_id 数量不足以抽取 {sample_count} 个 (pool size: {len(pool)})"
                )

        # 常规情形：对每个种子进行随机抽样
        for seed in range(seeds):
            random.seed(seed)
            sample = random.sample(pool, sample_count)
            sample.sort()
            output_data[ratio_str][str(seed)] = sample
    
    # 将结果保存为 JSON 格式到 output_txt 文件中
    with open(output_txt, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"随机抽取结果已保存到 {output_txt}")

if __name__ == '__main__':
    # 输入文件：之前生成的 high_iou_image_ids.txt，每行存有一个 image_id
    high_iou_txt = '/path/to/filtered_image_ids.txt'
    # 输出文件：结果保存为 random_supervision_image_ids.txt
    output_txt = '/path/to/random_supervision_image_ids.txt'

    generate_random_image_selection(high_iou_txt, output_txt)
