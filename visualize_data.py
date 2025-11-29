import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("mushroom.csv")

plt.style.use("ggplot")

# 1. Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df["class"])
plt.title("Class Distribution (Edible vs Poisonous)")
plt.savefig("viz_class_distribution.png")
plt.close()

# 2. Cap-shape distribution
plt.figure(figsize=(8,4))
sns.countplot(x=df["cap-shape"])
plt.title("Cap Shape Distribution")
plt.savefig("viz_cap_shape.png")
plt.close()

# 3. Habitat distribution
plt.figure(figsize=(8,4))
sns.countplot(x=df["habitat"])
plt.title("Habitat Distribution")
plt.savefig("viz_habitat.png")
plt.close()

# 4. Numeric Feature Histograms
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.savefig(f"viz_{col}.png")
    plt.close()

# 5. Correlation Heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig("viz_correlation_heatmap.png")
    plt.close()

print("Visualizations saved successfully!")
