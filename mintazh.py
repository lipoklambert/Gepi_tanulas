from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import numpy as np

#1. feladat
iris = load_iris();
n = iris.data.shape[0]; # number of records
p = iris.data.shape[1]; # number of attributes
k = iris.target_names.shape[0]; # number of target classes

# Printing the basic parameters
print(f'Number of records:{n}');
print(f'Number of attributes:{p}');
print(f'Number of target classes:{k}');

#2. feladat
iris_df = load_iris(as_frame=True);
sns.set(style="ticks");
sns.pairplot(iris_df.frame, hue="target");

#3.feladat
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2023)

#4.feladat
dt_model = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=2023)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)

lr_model = LogisticRegression(solver='liblinear', random_state=2023)
lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_test, y_test)

nn_model = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', max_iter=1000, random_state=2023)
nn_model.fit(X_train, y_train)
nn_score = nn_model.score(X_test, y_test)

print(f'\nDecision Tree Test Score: {dt_score}')
print(f'Logistic Regression Test Score: {lr_score}')
print(f'Neural Network Test Score: {nn_score}')

#5.feladat
best_model = dt_model  

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

#6.feladat
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#7.feladat
k_values = list(range(2, 31))
db_scores = []

for k in k_values:
    kmeans_model = KMeans(n_clusters=k, random_state=2023)
    kmeans_model.fit(X)

    # Predikciók elvégzése
    labels = kmeans_model.labels_

    # PCA használata a klaszterezés vizualizálásához
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # Davies-Bouldin index számolása és mentése
    db_score = davies_bouldin_score(X, labels)
    db_scores.append(db_score)

# Az optimális klaszterszám meghatározása
optimal_k = k_values[np.argmin(db_scores)]

print(f'Optimal number of clusters: {optimal_k}')

# K-közép modell újra tanítása az optimális klaszterszámmal
optimal_kmeans_model = KMeans(n_clusters=optimal_k, random_state=2023)
optimal_kmeans_model.fit(X)

# Predikciók elvégzése az optimal_kmeans_model segítségével
optimal_labels = optimal_kmeans_model.labels_

# PCA használata a klaszterezés vizualizálásához
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Klaszterek vizualizálása pontdiagrammon
plt.figure(figsize=(10, 8))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=optimal_labels, palette="Set2", s=50)
plt.title(f'K-means Clustering (K={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()