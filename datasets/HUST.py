import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

# ----------------------- HUST 严重故障-速度迁移 数据集配置 -----------------------
signal_size = 1024
speed_dict = {
    # 0: "20Hz", 1: "25Hz", 2: "30Hz", 3: "35Hz", 4: "40Hz",
    # 5: "60Hz", 6: "65Hz", 7: "70Hz", 8: "75Hz", 9: "80Hz",
    # 10: "VS_0_40_0Hz"
    0: "30Hz",
    1: "40Hz",
    2: "60Hz", 
    3: "VS_0_40_0Hz"
}

MEDIUM_PREFIX = "0.5X"  # 用于排除中等故障文件
state_codes = ["H", "I", "B", "O", "C"]
label_map = {code: k for k, code in enumerate(state_codes)}
MIN_SIGNAL_LENGTH = 2048  # 至少要有这么多点才认为文件有效


def parse_filename(file_name):
    """
    解析文件名，返回 (severity, fault_type, speed)
    """
    if file_name.lower().endswith(('.xlsx', '.xls')):
        name = file_name[:-5] if file_name.lower().endswith('.xlsx') else file_name[:-4]
    else:
        return None, None, None

    name_clean = name.replace(" ", "").upper()

    if "VS_0_40_0HZ" in name_clean:
        speed = "VS_0_40_0Hz"
        prefix_end = name_clean.find("VS_0_40_0HZ")
        prefix = name[:prefix_end].rstrip('_')
    else:
        parts = name_clean.split('_')
        speed_part = parts[-1]
        speed = speed_part.replace("HZ", "Hz")
        prefix = "_".join(parts[:-1])

    prefix_parts = prefix.split('_') if prefix else []
    if len(prefix_parts) == 0:
        return None, None, None
    elif len(prefix_parts) == 1:
        severity = "None"
        fault_type = prefix_parts[0]
    else:
        severity = prefix_parts[0]
        fault_type = prefix_parts[-1]

    if fault_type not in state_codes:
        return None, None, None

    return severity, fault_type, speed


def get_hust_speed_files_grouped(root, speed_ids):

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"HUST data directory does not exist: {root!r}. "
            "Pass the correct path with --data_dir."
        )
    target_speeds = [speed_dict[i] for i in speed_ids]
    print(f"\n>>> 加载 HUST 数据 (按文件级) | 目标转速: {target_speeds}")

    file_level_data = []

    for file_name in sorted(os.listdir(root)):
        severity, fault_type, speed = parse_filename(file_name)
        if speed is None or fault_type is None or speed not in target_speeds:
            continue

        if severity == MEDIUM_PREFIX:
            continue
        if fault_type == "H":
            pass
        elif severity != "None":
            continue

        file_path = os.path.join(root, file_name)
        try:
            vibration_data = []
            success_encoding = None
            for encoding in ['utf-8', 'gbk', 'latin1']:
                vibration_data.clear()
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        parts = line.split()
                        if len(parts) < 3: continue
                        try:
                            float(parts[0])
                            vibration_data.append(float(parts[2]))
                        except ValueError:
                            pass
                if len(vibration_data) >= MIN_SIGNAL_LENGTH:
                    success_encoding = encoding
                    break

            vibration_data = np.array(vibration_data, dtype=np.float32)
            if len(vibration_data) < MIN_SIGNAL_LENGTH:
                continue

            label = label_map[fault_type]
            file_level_data.append({
                "data": vibration_data,
                "label": label,
                "file_name": file_name
            })
            print(f"    ✓ 读取成功: {file_name} | 类别: {fault_type} | {len(vibration_data)} 点")

        except Exception as e:
            print(f"    × 读取失败: {file_name} → {e}")

    return file_level_data


def sliding_window(data_array, label, window_size, stride):
    """独立执行滑动窗口的辅助函数"""
    segments, labels = [], []
    start = 0
    while start + window_size <= len(data_array):
        segments.append(data_array[start : start + window_size])
        labels.append(label)
        start += stride
    return segments, labels


class HUST(object):
    num_classes = 5
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([Reshape(), Normalize(self.normlizetype), Retype()]),
            'val': Compose([Reshape(), Normalize(self.normlizetype), Retype()])
        }

    def _process_domain_data(self, file_data, val_ratio=0.2, stride=None):

        if stride is None:
            stride = signal_size
            
        train_seg, train_lab = [], []
        val_seg, val_lab = [], []

        for item in file_data:
            data = item["data"]
            label = item["label"]

            # 按时间戳比例严格切断序列
            split_idx = int(len(data) * (1 - val_ratio))
            train_series = data[:split_idx]
            val_series = data[split_idx:]

            # 分别对纯净的训练/验证序列进行滑动窗口。验证集建议 stride=signal_size（不重叠）
            t_seg, t_lab = sliding_window(train_series, label, signal_size, stride)
            v_seg, v_lab = sliding_window(val_series, label, signal_size, signal_size) 

            train_seg.extend(t_seg)
            train_lab.extend(t_lab)
            val_seg.extend(v_seg)
            val_lab.extend(v_lab)

        train_pd = pd.DataFrame({"data": train_seg, "label": train_lab})
        val_pd = pd.DataFrame({"data": val_seg, "label": val_lab})
        return train_pd, val_pd

    def data_split(self, transfer_learning=True, k_shot=None, k_shot_target=None, target_val_size=200):
        # ===================== 源域 =====================
        source_files = get_hust_speed_files_grouped(self.data_dir, self.source_N)
        train_pd, val_pd = self._process_domain_data(source_files, val_ratio=0.2, stride=signal_size)

        print("\n=== 源域数据划分 ===")
        print(f"总训练切片: {len(train_pd)} | 总验证切片: {len(val_pd)}")


        if k_shot is not None:
            few_list = []
            for c in range(self.num_classes):
                class_df = train_pd[train_pd["label"] == c]
                if len(class_df) > 0:
                    few_list.append(class_df.sample(n=min(k_shot, len(class_df)), random_state=42))
            train_pd = pd.concat(few_list, ignore_index=True) if few_list else pd.DataFrame(columns=["data", "label"])
            print(f"-> K-shot 降采样后源域训练集大小: {len(train_pd)}")

        source_train = dataset(train_pd, transform=self.data_transforms['train'])
        source_val = dataset(val_pd, transform=self.data_transforms['val'])

        # ===================== 目标域 =====================
        target_train, target_val = None, None
        if transfer_learning:
            target_files = get_hust_speed_files_grouped(self.data_dir, self.target_N)
            target_train_pd, target_val_pd = self._process_domain_data(target_files, val_ratio=0.2, stride=signal_size)

            print("\n=== 目标域数据划分 ===")
            print(f"目标域初始训练切片: {len(target_train_pd)} | 初始验证切片: {len(target_val_pd)}")

            # 限制目标域验证集大小 (从无污染的 target_val_pd 中均匀随机抽取)
            if target_val_size < len(target_val_pd):
                target_val_pd = target_val_pd.sample(n=target_val_size, random_state=42).reset_index(drop=True)

            # 目标域 K-shot
            if k_shot_target is not None:
                few_list = []
                for c in range(self.num_classes):
                    class_df = target_train_pd[target_train_pd["label"] == c]
                    if len(class_df) > 0:
                        few_list.append(class_df.sample(n=min(k_shot_target, len(class_df)), random_state=42))
                target_train_pd = pd.concat(few_list, ignore_index=True) if few_list else pd.DataFrame(columns=["data", "label"])
                print(f"-> K-shot 降采样后目标域训练集大小: {len(target_train_pd)}")

            target_train = dataset(target_train_pd, transform=self.data_transforms['train'])
            target_val = dataset(target_val_pd, transform=self.data_transforms['val'])

            return source_train, source_val, target_train, target_val
        else:
            return source_train, source_val, source_val
