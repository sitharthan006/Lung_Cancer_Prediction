import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

data = pd.read_csv(
    "data/processed_data/model_data.csv"
)


# Create output directory
os.makedirs('../images/eda_images', exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 1. Target Distribution (Final_Prediction)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot
target_counts = data['Final_Prediction'].value_counts()
colors = ['#2ecc71', '#e74c3c']
sns.barplot(x=target_counts.index, y=target_counts.values, palette=colors, ax=axes[0])
axes[0].set_title('Final Prediction Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('Count')
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# Pie chart
axes[1].pie(target_counts.values, labels=target_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Class Balance', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/eda_images/Target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 2. Correlation Heatmap
# ============================================================
numerical_data = data.select_dtypes(include=[np.number])
corr = numerical_data.corr()
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            linewidths=0.5, center=0, square=True,
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap (Numerical Features)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/eda_images/Correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 3. Top Correlated Features with Mortality_Risk
# ============================================================
# Since Final_Prediction is categorical, correlate with Mortality_Risk as proxy
top_corr = corr['Mortality_Risk'].drop('Mortality_Risk').sort_values(key=abs, ascending=False).head(10)
plt.figure(figsize=(10, 7))
colors_corr = ['#e74c3c' if x > 0 else '#3498db' for x in top_corr.values]
sns.barplot(x=top_corr.values, y=top_corr.index, palette=colors_corr)
plt.title('Top 10 Features Correlated with Mortality Risk', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('../images/eda_images/Top_correlated_features.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 4. Age Distribution by Prediction Outcome
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for outcome, color in zip(['Yes', 'No'], ['#e74c3c', '#2ecc71']):
    subset = data[data['Final_Prediction'] == outcome]['Age']
    axes[0].hist(subset, bins=40, alpha=0.6, label=f'Cancer: {outcome}', color=color, edgecolor='white')
axes[0].set_title('Age Distribution by Prediction', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].legend()

sns.kdeplot(data=data, x='Age', hue='Final_Prediction', palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
            fill=True, alpha=0.4, ax=axes[1])
axes[1].set_title('Age Density by Prediction', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/eda_images/Age_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 5. Smoking Status vs Final Prediction
# ============================================================
plt.figure(figsize=(12, 6))
ct = pd.crosstab(data['Smoking_Status'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.7)
plt.title('Smoking Status vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Smoking_Status vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 6. Gender vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Gender'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Gender vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Gender vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 7. Air Pollution Exposure vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Air_Pollution_Exposure'], data['Final_Prediction'])
ct.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Air Pollution Exposure vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Air Pollution Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Air_Pollution vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 8. Stage at Diagnosis vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Stage_at_Diagnosis'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.7)
plt.title('Stage at Diagnosis vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Stage')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Stage_at_Diagnosis vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 9. Cancer Type vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Cancer_Type'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Cancer Type vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Cancer_Type vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 10. Mutation Type vs Final Prediction
# ============================================================
if 'Mutation_Type' in data.columns:
    plt.figure(figsize=(12, 6))
    ct = pd.crosstab(data['Mutation_Type'], data['Final_Prediction'])
    ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.7)
    plt.title('Mutation Type vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Mutation Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Cancer Prediction')
    plt.tight_layout()
    plt.savefig('../images/eda_images/Mutation_Type vs Final_Prediction.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 11. Mortality Risk Distribution by Outcome
# ============================================================
plt.figure(figsize=(12, 6))
sns.kdeplot(data=data, x='Mortality_Risk', hue='Final_Prediction',
            palette={'Yes': '#e74c3c', 'No': '#2ecc71'}, fill=True, alpha=0.4)
plt.title('Mortality Risk Distribution by Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Mortality Risk')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('../images/eda_images/Mortality_Risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 12. 5-Year Survival Probability by Outcome
# ============================================================
plt.figure(figsize=(12, 6))
sns.kdeplot(data=data, x='5_Year_Survival_Probability', hue='Final_Prediction',
            palette={'Yes': '#e74c3c', 'No': '#2ecc71'}, fill=True, alpha=0.4)
plt.title('5-Year Survival Probability by Prediction', fontsize=14, fontweight='bold')
plt.xlabel('5-Year Survival Probability')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('../images/eda_images/Survival_Probability_distribution.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 13. Healthcare Access vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Healthcare_Access'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Healthcare Access vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Healthcare Access')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Healthcare_Access vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 14. Socioeconomic Status vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Socioeconomic_Status'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Socioeconomic Status vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Socioeconomic Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Socioeconomic_Status vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 15. Rural vs Urban vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Rural_or_Urban'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Rural vs Urban vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Location Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Rural_Urban vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 16. Family History vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Family_History'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Family History vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Family_History vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 17. Treatment Access vs Final Prediction
# ============================================================
plt.figure(figsize=(10, 6))
ct = pd.crosstab(data['Treatment_Access'], data['Final_Prediction'])
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.6)
plt.title('Treatment Access vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Treatment Access')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Cancer Prediction')
plt.tight_layout()
plt.savefig('../images/eda_images/Treatment_Access vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 18. Pairplot of Top Numerical Features
# ============================================================
# Sample for performance (pairplot is slow on large data)
sample = data.sample(n=min(5000, len(data)), random_state=42)
top_numerical = ['Age', 'Mortality_Risk', '5_Year_Survival_Probability', 'Risk_Score', 'Final_Prediction']
sns.pairplot(sample[top_numerical], hue='Final_Prediction',
             palette={'Yes': '#e74c3c', 'No': '#2ecc71'},
             diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})
plt.suptitle('Pairplot of Key Numerical Features', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/eda_images/Pairplot.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 19. Age vs Mortality Risk (Scatter)
# ============================================================
plt.figure(figsize=(12, 6))
sample_scatter = data.sample(n=min(10000, len(data)), random_state=42)
sns.scatterplot(data=sample_scatter, x='Age', y='Mortality_Risk', hue='Final_Prediction',
                palette={'Yes': '#e74c3c', 'No': '#2ecc71'}, alpha=0.4, s=15)
plt.title('Age vs Mortality Risk by Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Mortality Risk')
plt.tight_layout()
plt.savefig('../images/eda_images/Age vs Mortality_Risk.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 20. Exposure Count vs Final Prediction (Box Plot)
# ============================================================
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Final_Prediction', y='Exposure_Count',
            palette={'Yes': '#e74c3c', 'No': '#2ecc71'})
plt.title('Exposure Count vs Lung Cancer Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Prediction')
plt.ylabel('Number of Exposure Factors')
plt.tight_layout()
plt.savefig('../images/eda_images/Exposure_Count vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 21. Risk Score Distribution by Outcome
# ============================================================
plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x='Final_Prediction', y='Risk_Score',
               palette={'Yes': '#e74c3c', 'No': '#2ecc71'}, inner='box')
plt.title('Risk Score Distribution by Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Prediction')
plt.ylabel('Risk Score')
plt.tight_layout()
plt.savefig('../images/eda_images/Risk_Score vs Final_Prediction.png', dpi=150, bbox_inches='tight')
plt.close()

