# ml_quickstart.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1) Datos sintéticos
X, y = make_classification(
    n_samples=800, n_features=6, n_informative=4, n_redundant=0,
    random_state=42, class_sep=1.2
)
cols = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df["target"] = y

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    df[cols], df["target"], test_size=0.25, random_state=42, stratify=df["target"]
)

# 3) Modelo
clf = LogisticRegression(max_iter=200, n_jobs=None)
clf.fit(X_train, y_train)

# 4) Evaluación
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("Classification report:\n", classification_report(y_test, y_pred))

# 5) Visual rápido: matriz de confusión
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de Confusión")
plt.colorbar()
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()

def quick_ml():
    return "Versión 2 - Modelo actualizado listo para pruebas"

if __name__ == "__main__":
    print(quick_ml())