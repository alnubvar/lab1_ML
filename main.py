import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------
# Загрузка данных и подготовка
# ---------------------------
df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")
df["Target"] = (df["Solidity"] <= 0.9903).astype(int)
feat = [col for col in df.columns if col not in ["Class", "Target"]]

print("Начальные строки датасета:")
print(df.head())
print("\nИнфо о данных:")
print(df.info())
print("\nРаспределение классов:")
print(df["Target"].value_counts())

# ---------------------------
# Создаем папки для графиков
# ---------------------------
for i in range(1, 5):
    os.makedirs(f"EDA_graphs/Task{i}", exist_ok=True)

# ---------------------------
# Задание 1. Гистограммы
# ---------------------------
for col in feat:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue="Target", stat="probability", common_norm=False, bins=30, alpha=0.5)
    plt.title(f"Распределение признака {col} по классам")
    plt.tight_layout()
    plt.savefig(f"EDA_graphs/Task1/hist_{col}.png")
    plt.close()

# ---------------------------
# Задание 2. Моды
# ---------------------------
multimodal_feat = [col for col in feat if len(df[col].mode()) > 1]

for col in multimodal_feat:
    print(f"Признак '{col}' многомодальный, моды: {df[col].mode().values}")

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue="Target", stat="probability", common_norm=False, bins=30, alpha=0.5)
    plt.title(f"Многомодальная гистограмма {col}")
    plt.savefig(f"EDA_graphs/Task2/hist_multimodal_{col}.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.violinplot(x="Target", y=col, data=df, inner="quartile")
    plt.title(f"Violinplot {col} по Target")
    plt.savefig(f"EDA_graphs/Task2/violin_{col}.png")
    plt.close()

# ---------------------------
# Задание 3. Дискретная компонента
# ---------------------------
for col in feat:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} уникальных значений (из {len(df)})")

    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=50, color="skyblue")
    plt.title(f"Дискретная компонента {col}")
    plt.xlabel(col)
    plt.ylabel("Количество объектов")
    plt.tight_layout()
    plt.savefig(f"EDA_graphs/Task3/discrete_{col}.png")
    plt.close()

# ---------------------------
# Задание 4. Другие графики
# ---------------------------

# 1. Круговая диаграмма Target
df["Target"].value_counts().plot.pie(autopct="%1.1f%%", figsize=(5,5))
plt.title("Распределение целевого признака")
plt.ylabel("")
plt.savefig("EDA_graphs/Task4/pie_target.png")
plt.close()

# 2. Столбчатая диаграмма Target
sns.countplot(x="Target", data=df, palette="Set2")
plt.title("Столбчатая диаграмма Target")
plt.savefig("EDA_graphs/Task4/bar_target.png")
plt.close()

# 3-4. Boxplots
for col in ["Area", "Perimeter"]:
    sns.boxplot(x="Target", y=col, data=df, palette="Set3")
    plt.title(f"Boxplot: {col} vs Target")
    plt.savefig(f"EDA_graphs/Task4/box_{col}.png")
    plt.close()

# 5-6. Scatterplots
sns.scatterplot(x="Area", y="Perimeter", hue="Target", data=df, alpha=0.7)
plt.title("Scatterplot: Area vs Perimeter")
plt.savefig("EDA_graphs/Task4/scatter_area_perimeter.png")
plt.close()

sns.scatterplot(x="Compactness", y="Solidity", hue="Target", data=df, alpha=0.7)
plt.title("Scatterplot: Compactness vs Solidity")
plt.savefig("EDA_graphs/Task4/scatter_compactness_solidity.png")
plt.close()

# 7. Корреляционная матрица с аннотацией
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Корреляционная матрица признаков (аннотированная)")
plt.savefig("EDA_graphs/Task4/heatmap_corr.png")
plt.close()

# 8. Pairplot ключевых признаков
sns.pairplot(df[["Area", "Perimeter", "Solidity", "Target"]], hue="Target", corner=True)
plt.savefig("EDA_graphs/Task4/pairplot.png")
plt.close()

# 9. Violinplot Solidity
sns.violinplot(x="Target", y="Solidity", data=df, inner="quartile")
plt.title("Violinplot: Solidity vs Target")
plt.savefig("EDA_graphs/Task4/violin_solidity.png")
plt.close()

# 10. KDE Aspect_Ration
plt.figure(figsize=(6,4))
sns.kdeplot(df[df["Target"]==0]["Aspect_Ration"], shade=True, label="Target=0")
sns.kdeplot(df[df["Target"]==1]["Aspect_Ration"], shade=True, label="Target=1")
plt.title("KDE: Aspect_Ration по классам")
plt.legend()
plt.savefig("EDA_graphs/Task4/kde_aspect_ratio.png")
plt.close()

print("Все графики построены и сохранены в папки 'EDA_graphs/Task1-Task4'.")
