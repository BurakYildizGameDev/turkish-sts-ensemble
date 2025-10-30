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
