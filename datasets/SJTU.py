import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

# ----------------------- SJTU 齿轮箱轴承数据集配置 -----------------------
signal_size = 2048
MIN_SIGNAL_LENGTH = 2048 
MAX_POINTS_PER_FILE = 1000000 

# SJTU 数据集的 4 种健康状态
state_folders = ["健康", "点蚀", "剥落", "磨损"]
label_map = {folder: k for k, folder in enumerate(state_folders)}

# -------------------------------------------------------------------------
# 配置工况字典
# -------------------------------------------------------------------------
condition_dict = {
    # 0: "1800-0",  
    0: "1800-5",  
    # 2: "1800-10", 
    # 3: "2700-0",  
    1: "2700-5",
    # 5: "2700-10",
    # 6: "3600-0",
    2: "3600-5",
    # 8: "3600-10"
}


def parse_filename(file_name):
    """
    解析 SJTU 的文件名，返回 (speed, load, condition)
    """
    if not file_name.lower().endswith('.txt'):
        return None, None, None

    name = file_name[:-4]
    parts = name.split('-')
    
    if len(parts) < 3:
        return None, None, None
        
    speed = parts[1]
    load = parts[2]
    
    # 将 "转速-负载" 组合作为匹配字典的 condition 字符串
    condition = f"{speed}-{load}" 
    
    return speed, load, condition


def get_sjtu_files(root, condition_ids, stride=None, channel=0):
    """
    加载 SJTU 数据集
    condition_ids: 包含目标工况 ID 的列表 (例如 [0, 1] 对应 1800-0 和 1800-5)
    channel: 选择传感器通道 (0 表示 12:00 方向，1 表示 3:00 方向)
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"SJTU data directory does not exist: {root!r}. "
            "Pass the correct path with --data_dir."
        )
    if stride is None:
        stride = signal_size

    data_all = []
    label_all = []
    

    target_conditions = [condition_dict[i] for i in condition_ids]

    print(f"\n>>> 加载 SJTU 数据 | 目标工况 (转速-负载): {target_conditions} | 使用通道: {channel}")

    for folder_name in state_folders:
        folder_path = os.path.join(root, folder_name)
        if not os.path.exists(folder_path):
            print(f"    × 警告: 找不到文件夹 {folder_path}")
            continue

        label = label_map[folder_name]

        for file_name in sorted(os.listdir(folder_path)):
            speed, load, condition = parse_filename(file_name)

            if condition is None:
                continue

            # 匹配目标工况组合列表
            if condition not in target_conditions:
                continue

            file_path = os.path.join(folder_path, file_name)
            print(f"  匹配 → {folder_name}/{file_name} | 类别: {folder_name} | 转速: {speed} | 负载: {load}")

            try:
                vibration_data = []
                encodings_to_try = ['utf-8', 'gbk', 'latin1']
                success_encoding = None
                
                for encoding in encodings_to_try:
                    vibration_data.clear()
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                            point_count = 0  
                            for line in f:

                                if point_count >= MAX_POINTS_PER_FILE:
                                    break 
                                    
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if len(parts) < 2:
                                    continue
                                try:
                                    # 读取指定通道的数据
                                    val = float(parts[channel])
                                    vibration_data.append(val)
                                    point_count += 1 
                                except ValueError:
                                    pass # 跳过非数字行
                        
                        if len(vibration_data) >= MIN_SIGNAL_LENGTH:
                            success_encoding = encoding
                            break
                    except Exception:
                        continue

                vibration_data = np.array(vibration_data, dtype=np.float32)

                if len(vibration_data) < MIN_SIGNAL_LENGTH:
                    print(f"    × 数据太短或为空：{len(vibration_data)}")
                    continue

                print(f"    ✓ 读取成功（编码: {success_encoding}）：{len(vibration_data)} 个点 (受限于 MAX_POINTS)")

            except Exception as e:
                print(f"    × 读取失败：{folder_name}/{file_name} → {str(e)}")
                continue

            # 滑动窗口切分
            start = 0
            while start + signal_size <= len(vibration_data):
                segment = vibration_data[start : start + signal_size]
                data_all.append(segment)
                label_all.append(label)
                start += stride

    print(f"  总切分样本数：{len(data_all)}")
    return [data_all, label_all]


class SJTU(object):
    num_classes = 4  
    inputchannel = 1 

    def __init__(self, data_dir, transfer_task, normlizetype="0-1", channel=0):
        self.data_dir = data_dir
        self.source_N = transfer_task[0] 
        self.target_N = transfer_task[1] 
        self.normlizetype = normlizetype
        self.channel = channel
        self.data_transforms = {
            'train': Compose([Reshape(), Normalize(self.normlizetype), Retype()]),
            'val': Compose([Reshape(), Normalize(self.normlizetype), Retype()])
        }

    def data_split(self, transfer_learning=True, k_shot=None, k_shot_target=None, target_val_size=200):
        # ===================== 源域 =====================
        list_data = get_sjtu_files(self.data_dir, self.source_N, channel=self.channel)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        print("\n=== 源域类别分布 ===")
        print(data_pd['label'].value_counts().sort_index() if not data_pd.empty else "空")
        print("===================\n")

        if k_shot is not None:
            few_list = []
            for c in range(self.num_classes):
                class_df = data_pd[data_pd["label"] == c]
                if len(class_df) > 0:
                    few_list.append(class_df.sample(n=min(k_shot, len(class_df)), random_state=42))
            data_pd = pd.concat(few_list, ignore_index=True) if few_list else pd.DataFrame()

        n_classes = len(data_pd['label'].unique()) if not data_pd.empty else 0
        test_size = max(n_classes, int(len(data_pd) * 0.2), 1)

        if len(data_pd) <= test_size:
            train_pd, val_pd = data_pd, data_pd.iloc[:0]
        else:
            train_pd, val_pd = train_test_split(
                data_pd,
                test_size=test_size,
                random_state=42,
                stratify=data_pd["label"] if n_classes > 1 else None
            )

        source_train = dataset(train_pd, transform=self.data_transforms['train'])
        source_val = dataset(val_pd, transform=self.data_transforms['val'])

        # ===================== 目标域 =====================
        list_data = get_sjtu_files(self.data_dir, self.target_N, channel=self.channel)
        full_target_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

        print("\n=== 目标域类别分布 ===")
        print(full_target_pd['label'].value_counts().sort_index() if not full_target_pd.empty else "空")
        print("=====================\n")

        if target_val_size >= len(full_target_pd):
            target_val_pd = full_target_pd.copy()
            target_train_pd = pd.DataFrame(columns=full_target_pd.columns)
        else:
            target_train_pd, target_val_pd = train_test_split(
                full_target_pd,
                test_size=target_val_size,
                random_state=42,
                stratify=full_target_pd["label"],
                shuffle=True
            )

        if transfer_learning and k_shot_target is not None:
            few_list = []
            for c in range(self.num_classes):
                class_df = target_train_pd[target_train_pd["label"] == c]
                if len(class_df) > 0:
                    few_list.append(class_df.sample(n=min(k_shot_target, len(class_df)), random_state=42))
            target_train_pd = pd.concat(few_list, ignore_index=True) if few_list else pd.DataFrame(columns=target_train_pd.columns)

        target_val = dataset(target_val_pd, transform=self.data_transforms['val'])
        target_train = dataset(target_train_pd, transform=self.data_transforms['train'])

        if transfer_learning:
            return source_train, source_val, target_train, target_val
        else:
            return source_train, source_val, target_val
