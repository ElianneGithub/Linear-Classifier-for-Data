#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:22:36 2021

@author: joelda
"""

from data import *
from GradDesc import *

import numpy as np
import matplotlib.pyplot as plt

""" Nous avons récupéré les données data que nous allons utiliser
 L’objet data étant un dictionnaire, nous pouvons récupérer les clef avec la méthode keys() """
    
#print(iris['data'])
#print(iris['target'])

print("Voici les clées de data = ")

print(data.keys())


""" On observe donc que l'objet data a deux clées : input et target """

""" Nous allons utiliser es deux tables de données séparément et les stocker dans les variable X et Y """
""" Nous allons utiliser également la fonction shape pour connaiître les dimentions de chaque table de données """

X = data['input']

print("X.shape : ")
print(X.shape)

Y = data['target']

print("Y.shape : ")

print(Y.shape)

""" Maintenant que nous avons visualisé globalement les données : taille, dimensions... nous pouvont commencer à les utiliser """

""" Exercices """

"""  1 : Programmons une fonction qui calcule le nombre d’erreurs commises par un classifieur linéaire.  """


def nombre_erreur(u_vec, S):
    u = u_vec.reshape(4,3)
    
    v = u_vec.reshape(4,3)
    
    
    x = S[:,:-1]
    y = S[:, -1]
    y_target = np.zeros((len(y),4), float)
    for i in range(len(y)):
        y_target[i, int(y[i])] =  1
        
    y_temp = x.dot(v.T)
    
    y_prediction = y_temp.argmax(axis = 1)
    return((y != y_prediction).sum())
    


""" 2 : Programmons une fonction qui calcule une loss pour ce classifieur (basée sur la log-
vraisemblance et le softmax) """

def softmax(v):
    return(((np.exp(v).T)/(np.exp(v).sum(axis = 1))).T)


def loss(u_vec, S):
    
    u = u_vec.reshape(4,3)
    
    
    
    v = u_vec.reshape(4,3)
    
    
    
    x = S[:,:-1]
    
    y = S[:, -1]
    
    y_target = np.zeros((len(y),4),float)
    for i in range(len(y)):
        y_target[i, int(y[i])] =  1
        
    y_temp = x.dot(v.T)
    y_temp2 = (softmax(y_temp)*y_target).sum(axis=1)
    score = -np.log(y_temp2).sum()
    return(score)

""" À présent que nous avons programmé les fonctions dont nous auront besoin, nous pouvons maintenant transformer les données
de sorte à pouvoir les adapter aux fonctions """

""" Transformons la sortie de y = data['target'] en un vecteur one-hot """

Y_target = np.zeros((len(Y),4),float)
for i in range(len(Y)):
    Y_target[i, Y[i]] =  1

print("Y_target : ")    

print(Y_target)

""" Ensuite nous allons combiner les données de la table input et le vecteur one-hot et regarder la dimension """

S = np.column_stack((X,Y))

print("S.shape : ")
print(S.shape)

""" Nous allons générer une variable V de nombres aléatoires de dimension 4,3 que nous allons ensuite redimensioner """

V = np.random.randn(4,3)

U = np.column_stack((V))

U = U.reshape(-1)

""" Nous allons créer la variable Y_temp qui sera le produit des données de input et de la transposée de V """

Y_temp = X.dot(V.T)

""" Nous allons lui appliquer la fonction softmax,multiplier par le vecteur one-hot et ensuite faire la somme """

Y_temp2 = (softmax(Y_temp)*Y_target).sum(axis=1)

""" Pour obtenir le score nous faisosn le logarithme puis la somme encore """

score = -np.log(Y_temp2).sum()

print("Voici le score =") 

print(score)

""" Pour notre prédiction nous récupérons les indices des valeurs maximales et donc les positions des scores les plus importants """ 

Y_prediction = Y_temp.argmax(axis = 1)


print("Voici la prédiction Y avec Y_temp = ")

print(Y_prediction)

print("Voici (Y != Y_prediction).sum() = ")

print((Y != Y_prediction).sum())

""" Maintenant que nous avons une prédiction, nous pouvont directement reprendre notre variable U et chercher grâce à la fonction loss
la perte effectuée par notre prédiction précédente  """

print("Voici loss(U, S) = ")

print(loss(U, S))

""" Nous allons donc voir avec la fonction nb_error le nombre d'erreur qui a été fait """

print("Voici nb_error(U, S) = ")

print(nombre_erreur(U, S))

""" 3 : Appliquons la descente de gradient à cette loss"""

""" Nous allons stocker les doonnées de la déscente de gradient appliquée à notre fonction loss dans data1 """

data1 = grad_desc_n(loss, S, 12, 1000, step = 0.02, x_0 = None)


print("Voici la déscente de gradient appliquée à la loss : ")

print(data1)


""" 4 : Nous pouvont finalement chercher la solution trouvée et le nombre d'erreur commises par cette solution en utilisant la descente de gradient redimensionnée """

U_solution = data1.reshape(3,4)

#print(U_solution)

print("Voici le résultat de la loss :  ")

print(loss(U_solution, S))


print("Voici le nombre d'erreur commises par la solution  : ")

print(nombre_erreur(U_solution, S))

""" En général, le nombre d'erreur se trouve entre 10 et 20 """

