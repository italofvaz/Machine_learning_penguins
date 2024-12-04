

import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import graphviz
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score

data = sns.load_dataset('penguins')

data.head(25)

""" - Atributos numéricos por espécie:"""

with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=data.drop(['sex', 'island'], axis=1), hue="species", palette="pastel")

""" - Sexo por espécie:"""

with sns.axes_style('whitegrid'):

  grafico = sns.countplot(data=data, x='sex', hue="species", palette="pastel")

""" - Ilha por espécie:"""

with sns.axes_style('whitegrid'):

  grafico = sns.countplot(data=data, x='island', hue="species", palette="pastel")

# Handling missing data in the database

nulos = data.isnull().sum()
print(nulos)

# In this case, let's replace the NaN values with the mean of the values
mean = data['body_mass_g'].mean()
print(mean)
data['body_mass_g'] = data['body_mass_g'].apply(lambda value: mean if pd.isnull(value) else value)

mean = data['bill_length_mm'].mean()
print(mean)
data['bill_length_mm'] = data['bill_length_mm'].apply(lambda value: mean if pd.isnull(value) else value)

mean = data['bill_depth_mm'].mean()
print(mean)
data['bill_depth_mm'] = data['bill_depth_mm'].apply(lambda value: mean if pd.isnull(value) else value)

mean = data['flipper_length_mm'].mean()
print(mean)
data['flipper_length_mm'] = data['flipper_length_mm'].apply(lambda value: mean if pd.isnull(value) else value)

count = data['sex'].value_counts()
most = count.idxmax()
data['sex'] = data['sex'].apply(lambda value: most if pd.isnull(value) else value)

nulos = data.isnull().sum()
print(nulos)

data['sex_m_nom'] = data['sex'].apply(lambda x : 1 if x == 'Male' else 0)
data['sex_f_nom'] = data['sex'].apply(lambda x : 1 if x == "Female" else 0)

"""Descarte as colunas categóricas originais e mantenha a variável resposta na primeira coluna do dataframe."""

data.head()

data = data[["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex_m_nom", "sex_f_nom"]]

"""Separe a base de dados em treino e teste utilizando uma proporção de 2/3 para treino e 1/3 para testes."""



predictors_train, predictors_test, target_train, target_test = train_test_split(
    data.drop(['species'], axis=1),
    data['species'],
    test_size=0.25,
    random_state=123
)

predictors_train.head()

target_train.head()

predictors_train.shape

predictors_test.shape

model = DecisionTreeClassifier()
model = model.fit(predictors_train, target_train)

model.__dict__

tree_data = tree.export_graphviz(model, out_file=None)
grap = graphviz.Source(tree_data)
grap

"""Calcule e visualize a **matriz de confusão** para o modelo de **árvore de decisão** treinado com os **dados de teste** (1/3). Comente os resultados."""

CM = confusion_matrix(target_test,
                      target_predicted,
                      labels = model.classes_)
CMd = ConfusionMatrixDisplay(confusion_matrix= CM,
                            display_labels = model.classes_)
CMd.plot()

target_predicted = model.predict(predictors_test)
confusion_matrix= confusion_matrix(target_test, target_predicted)
print(confusion_matrix)

"""**b.** Acurácia"""

acuracy = accuracy_score(target_test, target_predicted)
print(f"{round(100*acuracy, 2)}%")

"""Qual a espécie de um penguim com as seguintes características:

| island | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g | sex |
| --- | --- | --- | --- | --- | --- |
| Biscoe | 38.2 | 18.1 | 185.0 | 3950.0 | Male |
"""

pinguim = np.array([38.2, 18.1, 185.0, 3950.0, 1, 0])
especie = model.predict(pinguim.reshape(1,-1))
print(especie)

"""---"""
