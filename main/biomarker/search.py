import time
from Bio import Entrez
from Bio import Medline

Entrez.email = "xvlizhao@gmail.com"

brain_regions = [
    "Temporal", "Cerebelum", "Calcarine", "Precuneus", "Frontal",
    "Cingulum", "Parietal", "Thalamus", "Occipital"
]

# 清理微生物名称的空格并去重
microbes = list(set([m.strip() for m in [
    "Roseburia intestinalis","Roseburia hominis","Parabacteroides chongii","Parabacteroides faecis",
    "Parabacteroides timonensis","Ruminococcus torques", "Mediterraneibacter catenae",
    "Ruminococcus torques" ,"Butyricicoccus pullicaecorum","Butyricicoccus porcorum ",
    "Agathobaculum desmolans","Paraprevotella xylaniphila", "Paraprevotella xylaniphila",
    "Paraprevotella clara","Oribacterium sinus","Oribacterium parvum",
    "Oribacterium asaccharolyticum","Enterocloster homin","Lacrimispora indolis",
    "Kineothrix alysoides","Fusicatenibacter saccharivorans","Clostridium porci",
    "Lacrimispora amygdalina","Intestinimonas butyriciproducens","Intestinimonas timonensis",
    "Clostridium phoceensis"
]]))

def search_asd_studies(region, microbe):
    """专注搜索ASD领域的三重组合：脑区+微生物+ASD关键词"""
    asd_keywords = '(autism OR ASD OR "autism spectrum disorder")'
    query = f'({region}[Title/Abstract] AND {microbe}[Title/Abstract]) AND {asd_keywords}'  
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=2, sort="relevance")  
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"Error searching {region} & {microbe}: {e}")
        return []

def fetch_paper_details(id_list):
    """获取论文详细信息"""
    if not id_list:
        return []
    try:
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        return records
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []
# 以脑区为主轴遍历
for region in brain_regions:
    region_save_path = f"ASD_BrainRegion.txt"  
    
    for microbe in microbes:
        time.sleep(1.5)  
        paper_ids = search_asd_studies(region, microbe)
        
        if not paper_ids:
            continue  
        
        papers = fetch_paper_details(paper_ids)
        with open(region_save_path, 'a') as f:  
            f.write(f"\n### {region} & {microbe} ###\n")
            
            for paper in papers:
                title = paper.get("TI", "No title")
                pmid = paper.get("PMID", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "No URL"
                abstract = paper.get("AB", "No abstract available")[:500] + "..."  
                
               
                f.write(f"Region:{region}\nMicrobe:{microbe}\nTitle: {title}\nURL: {url}\nAbstract: {abstract}\n\n")
                print(f"Found in {region}: {title[:50]}...")
    
    print(f"Completed {region}. Saved to {region_save_path}")