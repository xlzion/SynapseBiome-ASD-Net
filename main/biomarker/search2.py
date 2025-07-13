import time
from Bio import Entrez
from Bio import Medline

Entrez.email = "xvlizhao@gmail.com"  
SAVE_FILE = "//home/yangzongxian/xlz/ASD_GCN/main/ASD_Results.txt"  
BRAIN_REGIONS = [
    "Temporal", "Cerebellum", "Calcarine", "Precuneus", 
    "Frontal", "Cingulum", "Parietal", "Thalamus", "Occipital"
]


MICROBES = [
    "Roseburia intestinalis","Roseburia hominis","Parabacteroides chongii","Parabacteroides faecis",
    "Parabacteroides timonensis","Ruminococcus torques", "Mediterraneibacter catenae",
    "Ruminococcus torques" ,"Butyricicoccus pullicaecorum","Butyricicoccus porcorum ",
    "Agathobaculum desmolans","Paraprevotella xylaniphila", "Paraprevotella xylaniphila",
    "Paraprevotella clara","Oribacterium sinus","Oribacterium parvum",
    "Oribacterium asaccharolyticum","Enterocloster homin","Lacrimispora indolis",
    "Kineothrix alysoides","Fusicatenibacter saccharivorans","Clostridium porci",
    "Lacrimispora amygdalina","Intestinimonas butyriciproducens","Intestinimonas timonensis",
    "Clostridium phoceensis"
]

def search_asd_studies(region, microbe):
    """分阶段简化查询语句"""
    query = (
        f'({region}[Title/Abstract] OR "brain"[Title/Abstract]) '  # 放宽脑区限制
        f'AND {microbe}[Title/Abstract] '
        'AND (autism OR ASD OR "autism spectrum disorder")'
        # 暂时移除人类研究限制：'AND ("human"[MeSH] OR "clinical trial"[PT])'
    )
    print(f"正在查询: {query}")  # 调试用
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3, sort="relevance")
        record = Entrez.read(handle)
        return record["IdList"]
    except Exception as e:
        print(f"检索错误: {str(e)}")
        return []

def fetch_paper_details(id_list):
    if not id_list:
        return []
    try:
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        return records
    except Exception as e:
        print(f"获取文献详情失败: {str(e)}")
        return []

# 初始化文件
with open(SAVE_FILE, "w") as f:
    f.write("ASD脑肠轴研究文献汇总\n\n")

# 主循环（添加进度提示）
for region in BRAIN_REGIONS:
    print(f"\n== 正在扫描脑区: {region} ==")
    
    for microbe in MICROBES:
        time.sleep(1)  # 遵守NCBI API速率限制
        
        try:
            ids = search_asd_studies(region, microbe)
            if not ids:
                print(f"  {microbe}: 无结果")
                continue
            
            papers = fetch_paper_details(ids)
            output = [f"\n### {region} & {microbe} ###"]
            
            for p in papers:
                title = p.get("TI", "无标题")
                pmid = p.get("PMID", "")
                output.append(f"标题: {title}\n链接: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n")
            
            # 写入文件
            with open(SAVE_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(output) + "\n")
            
            print(f"  {microbe}: 找到 {len(papers)} 篇")
        
        except Exception as e:
            print(f"处理 {microbe} 时发生错误: {str(e)}")

print(f"\n完成！结果已保存至: {SAVE_FILE}")