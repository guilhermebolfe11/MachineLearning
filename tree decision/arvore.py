import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

treinamento = pd.read_csv('data/train.csv')
teste = pd.read_csv('data/test.csv')

teste['Title'] = [nameStr[1].strip().split('.')[0] for nameStr in
teste['Name'].str.split(',')]
aggAgeByTitleT = teste[['Age','Title']].groupby(['Title']).mean().reset_index()
aggAgeByTitleT.columns = ['Title','Mean_Age']
teste_novo = pd.merge(teste, aggAgeByTitleT,on="Title")
teste_novo.loc[teste_novo['Age'].isnull(),'Age']=teste_novo[teste_novo['Age'].isnull()]['Mean_Age']
teste_novo.loc[411,"Age"] = 37
teste_novo.loc[teste_novo['Fare'].isnull(),'Fare'] = teste_novo[teste_novo['Pclass'] == 3]['Fare'].mean()
teste_novo['IsCabinDataEmpty'] = 0
teste_novo.loc[teste_novo['Cabin'].isnull(),'IsCabinDataEmpty'] = 1
teste_novo['FamilySize'] = teste_novo['SibSp'] + teste_novo['Parch'] + 1
treinamento['Title'] = [nameStr[1].strip().split('.')[0] for nameStr in
treinamento['Name'].str.split(',')]
aggAgeByTitleDS = treinamento[['Age','Title']].groupby(['Title']).mean().reset_index()
aggAgeByTitleDS.columns = ['Title','Mean_Age']
treinamento_novo = pd.merge(treinamento, aggAgeByTitleDS,on="Title")
treinamento_novo.loc[treinamento_novo['Age'].isnull(),'Age']=treinamento_novo[treinamento_novo['Age'].isnull()]['Mean_Age']
treinamento_novo['IsCabinDataEmpty'] = 0
treinamento_novo.loc[treinamento_novo['Cabin'].isnull(),'IsCabinDataEmpty'] = 1
treinamento_novo.loc[treinamento_novo['Embarked'].isnull(),'Embarked'] = 'S'
treinamento_novo['FamilySize'] = treinamento_novo['SibSp'] + treinamento_novo['Parch'] + 1

treinamento_novo = treinamento_novo.drop('Mean_Age',axis=1)
treinamento_novo = treinamento_novo.drop('Cabin',axis=1)
treinamento_novo = treinamento_novo.drop('Fare',axis=1)
treinamento_novo = treinamento_novo.drop('Ticket',axis=1)
treinamento_novo = treinamento_novo.drop('SibSp',axis=1)
treinamento_novo = treinamento_novo.drop('Parch',axis=1)
treinamento_novo = treinamento_novo.drop('Name',axis=1)
treinamento_novo = treinamento_novo.drop('PassengerId',axis=1)

treinamento_novo = treinamento_novo.replace({"Sex": {"male":0}})
treinamento_novo = treinamento_novo.replace({"Sex": {"female":1}})

treinamento_novo = treinamento_novo.replace({"Embarked": {"S":0}})
treinamento_novo = treinamento_novo.replace({"Embarked": {"Q":1}})
treinamento_novo = treinamento_novo.replace({"Embarked": {"C":2}})

treinamento_novo.loc[ treinamento_novo['Age'] <= 16, 'Age']= 0
treinamento_novo.loc[(treinamento_novo['Age'] > 16) & (treinamento_novo['Age'] <= 32), 'Age'] = 1
treinamento_novo.loc[(treinamento_novo['Age'] > 32) & (treinamento_novo['Age'] <= 48), 'Age'] = 2
treinamento_novo.loc[(treinamento_novo['Age'] > 48) & (treinamento_novo['Age'] <= 64), 'Age'] = 3
treinamento_novo.loc[ treinamento_novo['Age'] > 64, 'Age'] = 4

treinamento_novo.loc[ treinamento_novo['FamilySize'] <= 2, 'FamilySize']= 0
treinamento_novo.loc[(treinamento_novo['FamilySize'] > 2) & (treinamento_novo['FamilySize'] <=4), 'FamilySize'] = 1
treinamento_novo.loc[(treinamento_novo['FamilySize'] > 4) & (treinamento_novo['FamilySize'] <=8), 'FamilySize'] = 2
treinamento_novo.loc[ treinamento_novo['FamilySize'] > 8, 'FamilySize'] = 4

treinamento_novo['Title'] = treinamento_novo['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
treinamento_novo['Title'] = treinamento_novo['Title'].replace('Mlle', 'Miss')
treinamento_novo['Title'] = treinamento_novo['Title'].replace('Ms', 'Miss')
treinamento_novo['Title'] = treinamento_novo['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
treinamento_novo['Title'] = treinamento_novo['Title'].map(title_mapping)
treinamento_novo['Title'] = treinamento_novo['Title'].fillna(0)

teste_novo = teste_novo.drop('Mean_Age',axis=1)
teste_novo = teste_novo.drop('Cabin',axis=1)
teste_novo = teste_novo.drop('Fare',axis=1)
teste_novo = teste_novo.drop('Ticket',axis=1)
teste_novo = teste_novo.drop('SibSp',axis=1)
teste_novo = teste_novo.drop('Parch',axis=1)
teste_novo = teste_novo.drop('Name',axis=1)
teste_novo = teste_novo.drop('PassengerId',axis=1)

teste_novo = teste_novo.replace({"Sex": {"male":0}})
teste_novo = teste_novo.replace({"Sex": {"female":1}})

teste_novo = teste_novo.replace({"Embarked": {"S":0}})
teste_novo = teste_novo.replace({"Embarked": {"Q":1}})
teste_novo = teste_novo.replace({"Embarked": {"C":2}})

teste_novo.loc[ teste_novo['FamilySize'] <= 2, 'FamilySize']= 0
teste_novo.loc[(teste_novo['FamilySize'] > 2) & (teste_novo['FamilySize'] <=4), 'FamilySize'] = 1
teste_novo.loc[(teste_novo['FamilySize'] > 4) & (teste_novo['FamilySize'] <=8), 'FamilySize'] = 2
teste_novo.loc[ teste_novo['FamilySize'] > 8, 'FamilySize'] = 4

teste_novo.loc[ teste_novo['Age'] <= 16, 'Age']= 0
teste_novo.loc[(teste_novo['Age'] > 16) & (teste_novo['Age'] <= 32), 'Age'] = 1
teste_novo.loc[(teste_novo['Age'] > 32) & (teste_novo['Age'] <= 48), 'Age'] = 2
teste_novo.loc[(teste_novo['Age'] > 48) & (teste_novo['Age'] <= 64), 'Age'] = 3
teste_novo.loc[ teste_novo['Age'] > 64, 'Age'] = 4


teste_novo['Title'] = teste_novo['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
teste_novo['Title'] = teste_novo['Title'].replace('Mlle', 'Miss')
teste_novo['Title'] = teste_novo['Title'].replace('Ms', 'Miss')
teste_novo['Title'] = teste_novo['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
teste_novo['Title'] = teste_novo['Title'].map(title_mapping)
teste_novo['Title'] = teste_novo['Title'].fillna(0)

def col_classes(rows, col):
    res = treinamento_novo[col].unique()
    return res


def classes_counts(rows):
    survived = rows[rows["Survived"] == 1]
    died = rows[rows["Survived"] == 0]

    return {"Survived": survived.shape[0], "Dead": died.shape[0]}


def partition(rows, column, conditions):
    ret = [None] * len(conditions)

    for i in range(len(ret)):
        ret[i] = rows[rows[column] == conditions[i]]

    return ret


def entropy(rows):
    survived = rows[rows["Survived"] == 1]
    died = rows[rows["Survived"] == 0]

    survived_count = survived.shape[0]
    died_count = died.shape[0]

    # se só tem uma classe a entropia é 0
    if (survived_count == 0 or died_count == 0):
        return 0

    survived_p = survived_count / (survived_count + died_count)
    died_p = 1.0 - survived_p

    # se só tem uma classe a entropia é 0
    if (died_p == 0 or survived_p == 0):
        return 0

    ret = -survived_p * math.log(survived_p, 2) - died_p * math.log(died_p, 2)

    return ret


def gain(rows, column):
    # entropia do nodo pai
    e = entropy(rows)

    # classes
    conditions = col_classes(rows, column)
    # filhos separados por classe
    children = partition(rows, column, conditions)

    # remove a classe "Survived"
    children.pop()

    w = 0
    for c in children:
        ec = entropy(c)
        if ec > 0:
            w += c.shape[0] / rows.shape[0] * ec

    return e - w


def find_best_gain(rows):
    # [ganho, 'coluna']
    max_gain = [0.0, '']

    columns = rows.columns.tolist()
    # remove a classe "Survived"
    del columns[0]

    # encontra o com maior ganho
    for i in range(len(columns)):
        g = gain(rows, columns[i])

        if (g >= max_gain[0]):
            max_gain[0] = g
            max_gain[1] = columns[i]

    return max_gain


class Leaf:

    def __init__(self, rows):
        self.predictions = classes_counts(rows)

    def print(self, space, attr=""):
        print(space, 'Leaf', self.predictions, attr)

class Decision_Node:

    # column = 'nome da coluna'
     # branches = dict de filhos, onde a chave é a classe e o valor é o nodo: {1:Decision_Node_Object}
    def __init__(self, column, branches):
         self.column = column
         self.branches = branches

    def print(self, space="", attr=""):
        print(space, "Node -> ", self.column, attr)
        for k, v in self.branches.items():
            v.print(space + " ", str(k))


def tree(rows):
    # acha o melhor separador
    best = find_best_gain(rows)

    # criterio de parada para essa branch
    if best[0] == 0 or entropy(rows) == 0:
        return Leaf(rows)

    conditions = col_classes(rows, best[1])
    children = partition(rows, best[1], conditions)
    branches = {}

    for i in range(len(children)):
        # remove a coluna usada para separar
        aux = children[i].drop(best[1], axis=1)
        branches[conditions[i]] = tree(aux)

    return Decision_Node(best[1], branches)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    value = row[node.column]
    branch = node.branches[value]

    return classify(row, branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    try:
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    except:
        print(counts)
        return
    return probs


def teste(rows,arvore):
    total_errors = 0

    for index, row in rows.iterrows():
        res = classify(row, arvore)
        error = ''

        if res['Survived'] >= res['Dead']:
            expexted = "Survived"
            if row['Survived'] == 0:
                total_errors += 1
                error = 'Error'
                expexted = "Dead"
        else:
            expected = 'Dead'
            if row['Survived'] == 1:
                total_errors += 1
                error = 'Error'
                expected = 'Survived'

        print('Expected:', expected, print_leaf(res), error)

    error_percent = str((total_errors / rows.shape[0]) * 100) + "%"
    print('')
    print('Total de erros:', total_errors, 'Total de amostras:', rows.shape[0])
    print('Erro:', error_percent)

    return float(error_percent.replace("%", ""))


def compare(rows,arvore):
    total_errors = 0

    for index, row in rows.iterrows():
        res = classify(row, arvore)
        error = ''

        if res['Survived'] >= res['Dead']:
            expexted = "Survived"
            if row['Survived'] == 0:
                total_errors += 1
                error = 'Error'
                expexted = "Dead"
        else:
            expected = 'Dead'
            if row['Survived'] == 1:
                total_errors += 1
                error = 'Error'
                expected = 'Survived'

    error_percent = str((total_errors / rows.shape[0]) * 100) + "%"
    print('Total de erros:', total_errors, 'Total de amostras:', rows.shape[0])
    print('Erro:', error_percent)

    return float(error_percent.replace("%", "")), total_errors

arvore = tree(treinamento_novo)
teste(treinamento_novo,arvore)
arvore.print()

for index, row in teste_novo.iterrows():
    res = classify(row, arvore)
    print(print_leaf(res))