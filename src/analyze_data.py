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

    # 2. Cümle Uzunluk Analizi (Histogram)
    df['len_a'] = df['text_a'].astype(str).apply(lambda x: len(x.split()))
    df['len_b'] = df['text_b'].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['len_a'], color="skyblue", label="Sentence A", kde=True, alpha=0.5)
    sns.histplot(df['len_b'], color="orange", label="Sentence B", kde=True, alpha=0.5)
    plt.title("Sentence Length Distribution (Word Count)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/sentence_lengths.png", dpi=300)
    print(f"✅ Length analysis saved: {OUTPUT_DIR}/sentence_lengths.png")
    plt.close()

    # 3. Kelime Bulutu (Word Cloud)
    text_corpus = " ".join(df['text_a'].astype(str).tolist() + df['text_b'].astype(str).tolist())
    
    wordcloud = WordCloud(width=1600, height=800, background_color='white', colormap='viridis').generate(text_corpus)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Frequent Words in Dataset", fontsize=20)
    plt.savefig(f"{OUTPUT_DIR}/wordcloud.png", dpi=300)
    print(f"✅ Word cloud saved: {OUTPUT_DIR}/wordcloud.png")
    plt.close()

    # 4. Etiket Bazlı Uzunluk Farkı Analizi (Boxplot)
    df['len_diff'] = abs(df['len_a'] - df['len_b'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label_value', y='len_diff', data=df, palette="Set2")
    plt.xticks([0, 1], ['Not Similar (0)', 'Similar (1)'])
    plt.title("Sentence Length Differences by Label")
    plt.xlabel("Class")
    plt.ylabel("Word Count Difference (Absolute)")
    plt.savefig(f"{OUTPUT_DIR}/length_diff_boxplot.png", dpi=300)
    print(f"✅ Length difference analysis saved: {OUTPUT_DIR}/length_diff_boxplot.png")
    plt.close()

if __name__ == "__main__":
    analyze_dataset()
