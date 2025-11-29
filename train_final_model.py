import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("mushroom.csv")

# All feature columns except class
features = [
    'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
    'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
    'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
    'veil-type', 'veil-color', 'has-ring', 'ring-type',
    'spore-print-color', 'habitat', 'season'
]

X = df[features]
y = df["class"]

# Label encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Fill categorical missing values with mode
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Fill numeric missing values with median
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode categorical columns
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

# Random Forest model
model = RandomForestClassifier(n_estimators=300, random_state=42)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# Save model + label encoder
joblib.dump(pipeline, "final_mushroom_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("FINAL MODEL TRAINED SUCCESSFULLY!")
