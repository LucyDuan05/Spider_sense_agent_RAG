# -*- coding: utf-8 -*-
import numpy as np

def map_nfstream_to_cicids(flow, feature_names):
    """
    将 nfstream 的 flow 对象属性映射到模型需要的 78 个特征维度
    """
    # nfstream 默认提供的基础属性和统计属性
    # 注意：某些列在 nfstream 中可能需要手动计算或对应
    # 以下为部分核心映射示例，你需要确保顺序与你的 feature_names 完全一致
    
    # 获取 nfstream 转出的字典
    f = flow.to_dict()
    
    # 建立映射（NFStream 字段名 -> 你的特征列表名）
    # 注意：NFStream 使用的是毫秒/字节等基础单位
    raw_map = {
        'Destination Port': f.get('dst_port', 0),
        'Flow Duration': f.get('bidirectional_duration_ms', 0) * 1000, # 转为微秒
        'Total Fwd Packets': f.get('src2dst_packets', 0),
        'Total Backward Packets': f.get('dst2src_packets', 0),
        'Total Length of Fwd Packets': f.get('src2dst_bytes', 0),
        'Total Length of Bwd Packets': f.get('dst2src_bytes', 0),
        'Fwd Packet Length Max': f.get('src2dst_max_ps', 0),
        'Fwd Packet Length Min': f.get('src2dst_min_ps', 0),
        'Fwd Packet Length Mean': f.get('src2dst_mean_ps', 0),
        'Fwd Packet Length Std': f.get('src2dst_stddev_ps', 0),
        # ... (以此类推映射完 78 个特征)
    }

    # 为了保证绝对准确，我们按 feature_names 的顺序提取
    vec = []
    for name in feature_names:
        # 如果 raw_map 里没定义，默认补 0 (或者根据 nfstream 属性查找)
        val = raw_map.get(name, 0)
        vec.append(val)
        
    return np.array(vec, dtype=np.float32)