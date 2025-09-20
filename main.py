import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Загрузка данных

df = pd.read_excel("Pumpkin_Seeds_Dataset.xlsx")

# Создаем бинарный целевой признак
df["Target"] = (df["Solidity"] <= 0.9903).astype(int)

print("Начальные строки датасета:")
print(df.head())
print("\nИнфо о данных:")
print(df.info())
print("\nРаспределение классов:")
print(df["Target"].value_counts())

# Общий список признаков
feat = [col for col in df.columns if col not in ["Class", "Target"]]


# Задание 1. Гистограммы

os.makedirs("EDA_graphs/Task1", exist_ok=True)

for col in feat:
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=col,
        hue="Target",
        stat="probability",
        common_norm=False,
        bins=30,
        alpha=0.5,
    )
    plt.title(f"Распределение признака {col} по классам")
    plt.tight_layout()
    plt.savefig(f"EDA_graphs/Task1/hist_{col}.png")
    plt.close()


# Задание 2. Моды

os.makedirs("EDA_graphs/Task2", exist_ok=True)

multimodal_feat = []

for col in feat:
    mode_values = df[col].mode()
    if len(mode_values) > 1:
        multimodal_feat.append(col)
        print(f"Признак '{col}' многомодальный, моды: {mode_values.values}")

# Строим графики для многомодальных признаков
for col in multimodal_feat:
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=col,
        hue="Target",
        stat="probability",
        common_norm=False,
        bins=30,
        alpha=0.5,
    )
    plt.title(f"Многомодальная гистограмма признака {col}")
    plt.savefig(f"EDA_graphs/Task2/hist_multimodal_{col}.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.violinplot(x="Target", y=col, data=df, inner="quartile")
    plt.title(f"Violinplot {col} по классам Target")
    plt.savefig(f"EDA_graphs/Task2/violin_{col}.png")
    plt.close()


# Задание 3. Дискретная компонента

os.makedirs("EDA_graphs/Task3", exist_ok=True)

print("Количество уникальных значений признаков:")
for col in feat:
    n_unique = df[col].nunique()
    total = df.shape[0]
    print(f"{col}: {n_unique} уникальных значений (всего {total} объектов)")

    # Узкие гистограммы для поиска дискретности
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=50, kde=False, color="skyblue")
    plt.title(f"Дискретная компонента признака {col}")
    plt.xlabel(col)
    plt.ylabel("Количество объектов")
    plt.tight_layout()
    plt.savefig(f"EDA_graphs/Task3/discrete_{col}.png")
    plt.close()


# Задание 4. Другие графики

os.makedirs("EDA_graphs/Task4", exist_ok=True)

# 1. Круговая диаграмма распределения классов
df["Target"].value_counts().plot.pie(autopct="%1.1f%%", figsize=(5, 5))
plt.title("Распределение целевого признака Target")
plt.ylabel("")
plt.savefig("EDA_graphs/Task4/pie_target.png")
plt.close()

# 2. Столбчатая диаграмма распределения классов
sns.countplot(x="Target", data=df, palette="Set2")
plt.title("Столбчатая диаграмма распределения классов")
plt.savefig("EDA_graphs/Task4/bar_target.png")
plt.close()

# 3. Boxplot для признака Area
sns.boxplot(x="Target", y="Area", data=df, palette="Set3")
plt.title("Boxplot: Area vs Target")
plt.savefig("EDA_graphs/Task4/box_area.png")
plt.close()

# 4. Boxplot для признака Perimeter
sns.boxplot(x="Target", y="Perimeter", data=df, palette="Set3")
plt.title("Boxplot: Perimeter vs Target")
plt.savefig("EDA_graphs/Task4/box_perimeter.png")
plt.close()

# 5. Scatterplot: Area vs Perimeter
sns.scatterplot(x="Area", y="Perimeter", hue="Target", data=df, alpha=0.7)
plt.title("Scatterplot: Area vs Perimeter")
plt.savefig("EDA_graphs/Task4/scatter_area_perimeter.png")
plt.close()

# 6. Scatterplot: Compactness vs Solidity
sns.scatterplot(x="Compactness", y="Solidity", hue="Target", data=df, alpha=0.7)
plt.title("Scatterplot: Compactness vs Solidity")
plt.savefig("EDA_graphs/Task4/scatter_compactness_solidity.png")
plt.close()

# 7. Корреляционная матрица
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Корреляционная матрица признаков")
plt.savefig("EDA_graphs/Task4/heatmap_corr.png")
plt.close()

# 8. Pairplot (часть признаков)
sns.pairplot(df[["Area", "Perimeter", "Solidity", "Target"]], hue="Target")
plt.savefig("EDA_graphs/Task4/pairplot.png")
plt.close()

# 9. Violinplot: Solidity vs Target
sns.violinplot(x="Target", y="Solidity", data=df, inner="quartile")
plt.title("Violinplot: Solidity vs Target")
plt.savefig("EDA_graphs/Task4/violin_solidity.png")
plt.close()

# 10. KDE-график для Aspect_Ration
plt.figure(figsize=(6, 4))
sns.kdeplot(df[df["Target"] == 0]["Aspect_Ration"], shade=True, label="Target=0")
sns.kdeplot(df[df["Target"] == 1]["Aspect_Ration"], shade=True, label="Target=1")
plt.title("KDE-график: Aspect_Ration по классам")
plt.legend()
plt.savefig("EDA_graphs/Task4/kde_aspect_ratio.png")
plt.close()

print("Все графики построены и сохранены в папки 'EDA_graphs/Task1-Task4'.")
