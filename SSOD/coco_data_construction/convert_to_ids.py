import json

def match_ids(txt_file, json_file, output_txt_file):
    # 读取txt文件中的文件名
    with open(txt_file, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    
    # 读取coco格式的json文件
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 获取coco文件中的images部分
    images = coco_data["images"]
    
    # 存储匹配的id
    matched_ids = []
    
    # 遍历txt中的文件名，匹配对应的id
    for filename in filenames:
        # 查找coco中images部分的file_name匹配的项
        for image in images:
            if image["file_name"] == filename:
                matched_ids.append(image["id"])
                break
    
    # 将匹配到的id输出到新的txt文件
    with open(output_txt_file, 'w') as f:
        for img_id in matched_ids:
            f.write(f"{img_id}\n")

    print(f"Matched IDs have been saved to {output_txt_file}")

# 使用示例
txt_file = "/path/to/filtered_pseudo_labeled_filenames"  # 存储文件名的txt文件
json_file = "/path/to/instance_train_pseudo_labeled.json"  # COCO格式的json文件
output_txt_file = "/path/to/filtered_pseudo_labeled_image_ids"  # 输出匹配的id到新的txt文件

match_ids(txt_file, json_file, output_txt_file)
