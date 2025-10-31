import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np

# Ayarlar
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')
OUTPUT_DIR = "results/figures/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_dataset(csv_path="data/processed/semantic_dataset.csv"):
    print(f"📊 Analiz Başlıyor: {csv_path}")
    
    # Veri Yükleme
    if not os.path.exists(csv_path):
        print(f"❌ Dosya bulunamadı: {csv_path}. Lütfen önce veri setini oluşturun.")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text_a', 'text_b', 'label_value'])
    df['label_value'] = df['label_value'].astype(int)
    
    # 1. Sınıf Dağılımı (Pie Chart)
    plt.figure(figsize=(8, 8))
    counts = df['label_value'].value_counts()
    
    # Label mapping
    label_map = {0: 'Not Similar (Negative)', 1: 'Similar (Positive)'}
    labels = [label_map.get(idx, f"Class {idx}") for idx in counts.index]
    
    # Profesyonel Renkler (Makale için)
    colors = ['#ff9999', '#66b3ff']
    if len(counts) > 2:
        colors = sns.color_palette('pastel')[0:len(counts)]
        
    plt.pie(counts, labels=labels, colors=colors[:len(counts)], autopct='%1.1f%%', 
            startangle=90, pctdistance=0.85, textprops={'fontsize': 12})
            
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title("Dataset Class Balance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Class_balance.png", dpi=300)
    print(f"✅ Class balance saved: {OUTPUT_DIR}/Class_balance.png")
    plt.close()
