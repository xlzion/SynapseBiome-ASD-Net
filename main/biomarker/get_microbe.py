from adversarial import load_pretrained_mlp, extract_microbe_features
from MLP import MicrobeDataLoader, SparseMLP
import numpy as np
import pandas as pd
import torch
from biom import load_table
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class MicrobeDataLoader:
    def __init__(self, csv_path, biom_path, batch_size=32):
        
        self.batch_size = batch_size
        self.features_tensor, self.labels_tensor = self._load_from_files(csv_path, biom_path)
        
        # 添加数据验证
        assert len(self.features_tensor) > 0, "特征数据为空！"
        assert len(self.labels_tensor) > 0, "标签数据为空！"
        assert len(self.features_tensor) == len(self.labels_tensor), "特征与标签数量不匹配！"
        
        self._split_data()
        self.feature_names = self._get_feature_names(biom_path)
    
    
    def _get_feature_names(self, biom_path):
        """从BIOM文件中提取特征名称（含分类学信息）"""
        try:
            table = load_table(biom_path)
            
            
            feature_names = []
            for i, obs_id in enumerate(table.ids(axis='observation')):
                # 获取元数据中的分类信息
                metadata = table.metadata(obs_id, axis='observation')
                
                if metadata and 'taxonomy' in metadata:
                    # 示例格式：['k__Bacteria', 'p__Firmicutes', ...]
                    taxonomy = metadata['taxonomy']
                    
                    # 提取最具体分类等级（种级）
                    species = next((t for t in reversed(taxonomy) if t != ''), 'Unclassified')
                    feature_names.append(f"{species} (OTU-{obs_id})")
                else:
                    feature_names.append(f"Unclassified (OTU-{obs_id})")
                    
            return feature_names
        except Exception as e:
            print(f"特征名称获取失败: {str(e)}")
            return [f"Feature_{i}" for i in range(table.shape[0])]  # 生成默认名称
        
    def get_feature_names(self):
        
        return self.feature_names

 
    
    def _load_from_files(self, csv_path, biom_path):
        
        metadata = pd.read_csv(csv_path)
        #print(f"元数据加载成功，样本数: {len(metadata)}")
        
        table = load_table(biom_path)
        #print(f"BIOM表加载成功，原始样本数: {table.shape[1]}")

        sample_ids = set(table.ids(axis="sample"))
        metadata_ids = set(metadata["ID"])
        matched_ids = sample_ids & metadata_ids

        #print(f"匹配样本数: {len(matched_ids)}")
        unmatched = sample_ids - metadata_ids
        '''if unmatched:
            print(f"未匹配样本ID: {unmatched}")'''

        metadata = metadata[metadata["ID"].isin(matched_ids)].set_index("ID")
        filtered_table = table.filter(matched_ids, axis="sample", inplace=False)

        # 转换为DataFrame并验证
        df = filtered_table.to_dataframe(dense=True).T  # 明确指定dense格式
       
        feature_data = self._add_feature_engineering(filtered_table)
        feature_data = feature_data.T

        feature_data = df.values.astype(np.float32)
        return torch.tensor(feature_data, dtype=torch.float32), torch.tensor(metadata['DX_GROUP'].values, dtype=torch.long)
    
    def _add_feature_engineering(self, table):
        data = table.matrix_data.T.toarray()
        return np.log1p(data)
    
    def _split_data(self):
        try:
            train_features, test_features, train_labels, test_labels = train_test_split(
                self.features_tensor, self.labels_tensor, test_size=0.2, random_state=42
            )
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=0.1, random_state=42
            )

            self.train_dataset = TensorDataset(train_features, train_labels)
            self.val_dataset = TensorDataset(val_features, val_labels)
            self.test_dataset = TensorDataset(test_features, test_labels)
        except Exception as e:
            print(f"数据划分失败: {str(e)}")
            raise

    def get_loaders(self):
        return (
            DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4),
            DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4),
            DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        )


def get_top_microbes(model, feature_names, top_k=100):
    """
    获取微生物组中最重要的前top_k个特征
    原理：分析输入层权重绝对值之和
    """
    # 获取第一个全连接层的权重
    first_layer = model.feature_extractor[0]
    weights = first_layer.weight.data.cpu().numpy()
    
    # 计算每个特征的绝对权重之和
    importance_scores = np.abs(weights).sum(axis=0)
    
    # 获取排序后的索引
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    # 输出结果
    print("\n微生物组重要特征 Top", top_k)
    for i in range(top_k):
        idx = sorted_indices[i]
        print(f"{i+1}. {feature_names[idx]} ({importance_scores[idx]:.4f})")


csv_path = "/home/yangzongxian/xlz/ASD_GCN/microbe/data2/microbe_data.csv"
biom_path = "/home/yangzongxian/xlz/ASD_GCN/microbe/data2/feature-table.biom"
mlp_model = load_pretrained_mlp()
microbe_loader = MicrobeDataLoader(csv_path = csv_path, biom_path=biom_path, batch_size=32)
microbe_train_loader = microbe_loader.get_loaders()[0]
microbe_val_loader = microbe_loader.get_loaders()[1]
microbe_test_loader = microbe_loader.get_loaders()[2]
microbe_train_features, microbe_train_labels = extract_microbe_features(microbe_train_loader)
microbe_val_features, microbe_val_labels = extract_microbe_features(microbe_val_loader)
microbe_test_features, microbe_test_labels = extract_microbe_features(microbe_test_loader)
# 假设microbe_features是特征名称列表（从biom文件获取）
microbe_loader = MicrobeDataLoader(csv_path, biom_path)
feature_names = microbe_loader.get_feature_names()
get_top_microbes(mlp_model, feature_names, top_k=100)
