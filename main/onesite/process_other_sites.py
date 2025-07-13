import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import os

def get_site_subjects(h5_file, site_name):
    """获取指定站点的所有受试者ID"""
    subjects = []
    for subject_id in h5_file['/patients'].keys():
        if h5_file[f'/patients/{subject_id}'].attrs['site'] == site_name:
            subjects.append(subject_id)
    return subjects

def create_leavesiteout_dataset(h5_file, site_name, derivative='cc200'):
    """为指定站点创建留一站点验证数据集，创建10个不同的划分组"""
    # 获取该站点的所有受试者
    site_subjects = get_site_subjects(h5_file, site_name)
    
    # 获取所有其他站点的受试者
    other_subjects = []
    for subject_id in h5_file['/patients'].keys():
        if h5_file[f'/patients/{subject_id}'].attrs['site'] != site_name:
            other_subjects.append(subject_id)
    
    # 创建新的数据集组
    group_path = f'/experiments/{derivative}_leavesiteout-{site_name}'
    if group_path in h5_file:
        del h5_file[group_path]
    
    group = h5_file.create_group(group_path)
    group.attrs['derivative'] = derivative
    
    # 创建10个不同的划分组
    for fold in range(10):
        # 使用不同的随机种子创建不同的划分
        random_seed = 42 + fold
        
        # 划分其他站点的数据
        train_subjects, temp_subjects = train_test_split(
            other_subjects, 
            test_size=0.3, 
            random_state=random_seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, 
            test_size=0.5, 
            random_state=random_seed
        )
        
        # 创建站点组
        site_group = group.create_group(str(fold))
        
        # 保存数据集划分
        site_group.create_dataset('train', data=np.array(train_subjects, dtype='S'))
        site_group.create_dataset('valid', data=np.array(val_subjects, dtype='S'))
        site_group.create_dataset('test', data=np.array(site_subjects, dtype='S'))
        
        print(f"站点 {site_name} 的第 {fold+1} 折划分完成：")
        print(f"训练集: {len(train_subjects)} 个样本")
        print(f"验证集: {len(val_subjects)} 个样本")
        print(f"测试集: {len(site_subjects)} 个样本")
        print("-" * 50)

def main():
    # 打开HDF5文件
    file_path = "/home/yangzongxian/xlz/ASD_GCN/main/data2/abide.hdf5"
    with h5py.File(file_path, 'a') as f:
        # 获取所有唯一的站点名称
        sites = set()
        for subject_id in f['/patients'].keys():
            site = f[f'/patients/{subject_id}'].attrs['site']
            sites.add(site)
        
        # 为每个站点创建留一站点验证数据集
        for site in sites:
            if site != 'NYU':  # 跳过NYU，因为它已经有处理好的数据
                print(f"\n处理站点: {site}")
                # 为每种特征类型创建数据集
                for derivative in ['aal', 'cc200', 'ez']:
                    print(f"\n处理特征类型: {derivative}")
                    create_leavesiteout_dataset(f, site, derivative)

if __name__ == "__main__":
    main() 