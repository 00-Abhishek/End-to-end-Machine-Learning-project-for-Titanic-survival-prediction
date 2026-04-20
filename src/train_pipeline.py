import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# -------------------- LOAD DATA --------------------
df = pd.read_csv('data/Titanic-Dataset.csv')

# -------------------- CLEAN --------------------
df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, errors='ignore')

# -------------------- FEATURES & TARGET --------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

# -------------------- COLUMN GROUPS --------------------
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
cat_cols = ['Sex', 'Embarked', 'Pclass']

# -------------------- PREPROCESSING --------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# -------------------- FULL PIPELINE --------------------
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- TRAIN --------------------
model.fit(X_train, y_train)

# -------------------- EVALUATE --------------------
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------- SAVE --------------------
with open('models/titanic_pipeline.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Pipeline saved successfully ✅")