import h5py

import h5py
import numpy as np

def get_basename(item):
    """获取组或数据集的基名称"""
    name = item.name
    if name == '/':
        return '/'
    return name.split('/')[-1]

def print_hdf5_structure(item, indent_level=0):
    """递归打印HDF5文件结构"""
    indent = '  ' * indent_level
    basename = get_basename(item)

    # 处理组
    if isinstance(item, h5py.Group):
        # 打印组名称
        if item.name == '/':
            print(f"{indent}Group: /")
        else:
            print(f"{indent}Group: {basename}")
        
        # 打印组的属性
        for attr_name in item.attrs:
            attr_value = item.attrs[attr_name]
            # 处理不同类型的属性值
            if isinstance(attr_value, np.ndarray):
                attr_str = f"ndarray{attr_value.shape} {attr_value.dtype}"
            elif isinstance(attr_value, bytes):
                attr_str = attr_value.decode()
            else:
                attr_str = str(attr_value)
            print(f"{indent}  Attribute: {attr_name}: {attr_str}")
        
        # 递归遍历子项
        for key in item:
            child = item[key]
            print_hdf5_structure(child, indent_level + 1)
    
    # 处理数据集
    elif isinstance(item, h5py.Dataset):
        # 打印数据集信息
        print(f"{indent}Dataset: {basename} (Shape: {item.shape}, Dtype: {item.dtype})")
        
        # 打印数据集的属性
        for attr_name in item.attrs:
            attr_value = item.attrs[attr_name]
            if isinstance(attr_value, np.ndarray):
                attr_str = f"ndarray{attr_value.shape} {attr_value.dtype}"
            elif isinstance(attr_value, bytes):
                attr_str = attr_value.decode()
            else:
                attr_str = str(attr_value)
            print(f"{indent}  Attribute: {attr_name}: {attr_str}")

# 示例用法
with h5py.File('/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5', 'r') as f:
    print_hdf5_structure(f)