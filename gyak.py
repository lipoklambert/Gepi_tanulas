from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1.feladat
cancer = load_breast_cancer()
n = cancer.data.shape[0]
p = cancer.data.shape[1]
k = cancer.target_names.shape[0]

print(f'Number of records:{n}')
print(f'Number of attributes:{p}')
print(f'Number of target classes:{k}')

#2.feladat
cancer = load_breast_cancer()
data = cancer.data
feature_names = cancer.feature_names

plt.figure(figsize=(12, 8))

plt.scatter(data[:, 0], data[:, 1], c=cancer.target, cmap='viridis', edgecolors='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Scatter Plot of the First Two Variables')

plt.show()

#3.feladat
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=2023)