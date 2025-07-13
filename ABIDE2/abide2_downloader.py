# master_abide2_downloader.py
# 一个用于下载和准备 ABIDE II BIDS 数据集的综合脚本

import argparse
import pandas as pd
import boto3
import os
import json
from botocore import UNSIGNED
from botocore.client import Config

# --- 全局配置 ---
BUCKET_NAME = 'fcp-indi'
BASE_PREFIX = 'data/Projects/ABIDE2/RawDataBIDS/'
# 扫描参数模板 (简化版，实际应用中应更详尽)
SCAN_PARAMS_TEMPLATE = {
    'func': {
        'RepetitionTime': 2.0, # 这是一个示例值，需要根据具体中心调整
        'TaskName': 'rest'
    }
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ABIDE II BIDS 数据集下载与整理工具')
    parser.add_argument('phenotypic_file', type=str, help='从 NITRC 下载的 ABIDE II 复合表型 CSV 文件路径')
    parser.add_argument('output_dir', type=str, help='下载和整理 BIDS 数据集的本地根目录')
    parser.add_argument('--min_age', type=float, default=None, help='被试最小年龄 (包含)')
    parser.add_argument('--max_age', type=float, default=None, help='被试最大年龄 (包含)')
    parser.add_argument('--sex', type=int, choices=[1, 2], default=None, help='被试性别 (1=男性, 2=女性)')
    parser.add_argument('--dx', type=int, choices=[1, 2], default=None, help='诊断组 (1=ASD, 2=TC)')
    parser.add_argument('--max_subjects', type=int, default=None, help='下载的最大被试数量 (用于测试)')
    return parser.parse_args()

def filter_cohort(pheno_file, args):
    """根据条件筛选被试队列"""
    print(f"正在从 {pheno_file} 筛选队列...")
    df = pd.read_csv(pheno_file, encoding='latin1')
    df.columns = df.columns.str.strip()
    
    if args.min_age:
        df = df[df['AGE_AT_SCAN'] >= args.min_age]
    if args.max_age:
        df = df[df['AGE_AT_SCAN'] <= args.max_age]
    if args.sex:
        df = df[df['SEX'] == args.sex]
    if args.dx:
        df = df[df['DX_GROUP'] == args.dx]
        
    if args.max_subjects:
        df = df.head(args.max_subjects)
        
    print(f"筛选出 {len(df)} 名被试。")
    return df

def download_data(subject_df, output_dir):
    """使用 boto3 下载数据"""
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    subject_ids = subject_df['SUB_ID'].tolist()
    
    for sub_id in subject_ids:
        print(f"\n--- 处理被试: {sub_id} ---")
        subject_prefix = f"{BASE_PREFIX}sub-{sub_id}/"
        
        paginator = s3_client.get_paginator('list_objects_v2')
        try:
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=subject_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        local_path = os.path.join(output_dir, s3_key.replace(BASE_PREFIX, ''))
                        local_file_dir = os.path.dirname(local_path)
                        os.makedirs(local_file_dir, exist_ok=True)
                        
                        if not os.path.exists(local_path):
                            print(f"下载: {s3_key} -> {local_path}")
                            s3_client.download_file(BUCKET_NAME, s3_key, local_path)
                        else:
                            print(f"已存在, 跳过: {local_path}")
        except Exception as e:
            print(f"下载被试 {sub_id} 数据时出错: {e}")

def create_bids_metadata(subject_df, output_dir):
    """创建 BIDS 元数据文件 (participants.tsv 和 JSON sidecars)"""
    print("\n--- 创建 BIDS 元数据文件 ---")
    
    # 1. 创建 participants.tsv
    participants_df = subject_df.copy()
    participants_df['participant_id'] = "sub-" + participants_df['SUB_ID'].astype(str)
    participants_df = participants_df.rename(columns={'AGE_AT_SCAN': 'age', 'SEX': 'sex', 'DX_GROUP': 'group'})
    participants_df = participants_df[['participant_id', 'age', 'sex', 'group']]
    participants_df.fillna('n/a', inplace=True)
    participants_tsv_path = os.path.join(output_dir, 'participants.tsv')
    participants_df.to_csv(participants_tsv_path, sep='\t', index=False)
    print(f"已创建: {participants_tsv_path}")
    
    # 2. 创建 JSON sidecars
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                json_path = os.path.join(root, file.replace('.nii.gz', '.json'))
                if not os.path.exists(json_path):
                    if 'func' in root and '_bold' in file:
                        with open(json_path, 'w') as f:
                            json.dump(SCAN_PARAMS_TEMPLATE['func'], f, indent=4)
                        print(f"已创建模板: {json_path}")

def main():
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤 1: 筛选队列
    cohort_df = filter_cohort(args.phenotypic_file, args)
    
    # 步骤 2: 下载数据
    download_data(cohort_df, args.output_dir)
    
    # 步骤 3: 创建元数据
    create_bids_metadata(cohort_df, args.output_dir)
    
    print("\n--- 工作流程完成 ---")
    print("数据已下载并整理到 BIDS 结构中。")
    print("强烈建议您现在使用 'bids-validator' 工具验证数据集的完整性:")
    print(f"bids-validator {args.output_dir}")

if __name__ == '__main__':
    main()  