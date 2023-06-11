"""
    Auteur : Alessia Stefanelli - 464273
    Date : 07/04/2019
    Objet : Partie 4
"""
import numpy as np
import random


def sigmoid(x):
    """Cette fonction calcule la valeur de la fonction sigmoide (de 0 à 1) pour un nombre réel donné.
    Il est à noter que x peut également être un vecteur de type numpy array, dans ce cas, la valeur de la sigmoide correspondante à chaque réel du vecteur est calculée.
    En retour de la fonction, il peut donc y avoir un objet de type numpy.float64 ou un numpy array."""
    return 1 / (1 + np.exp(-x))

########   NOUVELLE FONCTIONS    #################

def ReLu(x):
    """
    Cette fonction calcule x en utilisant la fonction ReLU (rectified linear unit)
    :param x: nombre réel donné ou vecteur numpy array
    """
    return np.maximum(0,x)


def LeakyReLU(x):
    return np.maximum(0.01*x, x)



def SWISH(x):
    return x*sigmoid(x)


def FctTanHyp(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


###################################################


def initWeights(nb_rows, nb_columns):
    """Fonction destinée à initialiser les poids d'une matrice de genre (nb_rows * nb_columns) pour un réseau de neurones sur base d'une distribution normale de moyenne 0 et d'un écart-type de 0.0001."""
    return np.random.normal(0, 0.0001, (nb_rows, nb_columns))


def createNN(n_input, n_hidden):
    """Fonction permettant de créer un réseau de neurones en fonction de la taille de la couche d'entrée (n_input) et de la couche intermédiaire (n_hidden)
    Le réseau de neurones créé est ensuite retourné sous la forme d'un tuple de 2 numpy array contenant respectivement
    les coefficients de poids liant la couche d'entrée à la couche intermédiaire et les coefficients de poids liant la couche intermédiaire à la couche de sortie.
    """
    W_int = initWeights(n_hidden, n_input)
    W_out = initWeights(n_hidden, 1)[:,
            0]  # W2 est traité comme vecteur est non comme une matrice Hx1 (simplifie certaines ecritures)
    return (W_int, W_out)


def forwardPass(s, NN, act):
    """Cette fonction permet d'utiliser un réseau de neurones NN pour estimer la probabilité de victoire finale du joueur blanc pour un état (s) donné du jeu."""
    W_int = NN[0]
    W_out = NN[1]
    p_out = None

    ###### UTILISATION DES DIFFÉRENTES FONCTIONS D'ACTIVATION ######
    if act == "ReLU":
        P_int = ReLu(np.dot(W_int, s))
        p_out = ReLu(P_int.dot(W_out))
    elif act == "Sigmoid":
        P_int = sigmoid(np.dot(W_int, s))
        p_out = sigmoid(P_int.dot(W_out))
    elif act == "Leaky ReLU":
        P_int = LeakyReLU(np.dot(W_int, s))
        p_out = LeakyReLU(P_int.dot(W_out))
    elif act == "SWISH":
        P_int = SWISH(np.dot(W_int, s))
        p_out = SWISH(P_int.dot(W_out))
    elif act == "Fonction tangente hyperbolique":
        P_int = FctTanHyp(np.dot(W_int, s))
        p_out = FctTanHyp(P_int.dot(W_out))
    return p_out


def backpropagation(act, s, NN, delta, learning_strategy=None):
    """Fonction destinée à réaliser la mise à jour des poids d'un réseau de neurones (NN). Cette mise à jour se fait conformément à une stratégie d'apprentissage (learning_strategy)
    pour un état donné (s) du jeu.
    Le delta est la différence de probabilité de gain estimée entre deux états successif potentiels du jeu.
    La stratégie d'apprentissage peut soit être None, soit il s'agit d'un tuple de la forme ('Q-learning', alpha) où alpha est le learning_rate (une valeur entre 0 et 1 inclus),
    soit il s'agit d'un tuple de la forme ('TD-lambda', alpha, lamb, Z_int, Z_out) où alpha est le learning_rate, lamb est la valeur de lambda (entre 0 et 1 inclus) et
    Z_int et Z_out contiennent les valeurs de l'éligibility trace associées respectivement aux différents poids du réseau de neurones.
    La fonction de backpropagation ne retourne rien de particulier (None) mais les poids du réseau de neurone NN (W_int, W_out) peuvent être modifiés,
    idem pour l'eligibility trace (Z_int et Z_out) dans le cas où la stratégie TD-lambda est utilisée.
    """
    # remarque : les operateurs +=, -=, et *= changent directement un numpy array sans en faire une copie au prealable, ceci est nécessaire
    # lorsqu'on modifie W_int, W_out, Z_int, Z_out ci-dessous (sinon les changements seraient perdus après l'appel)
    if learning_strategy is None:
        # pas de mise à jour des poids
        return None
    W_int = NN[0]
    W_out = NN[1]

    ###### UTILISATION DES DIFFÉRENTES FONCTIONS D'ACTIVATION ######
    if act == "ReLU":
        P_int = ReLu(np.dot(W_int, s))
        p_out = ReLu(P_int.dot(W_out))
        grad_out = 1 if p_out >= 0 else 0
        grad_int = 1 * (P_int >= 0)
    elif act == "Sigmoid":
        P_int = sigmoid(np.dot(W_int, s))
        p_out = sigmoid(P_int.dot(W_out))
        grad_out = p_out * (1 - p_out)
        grad_int = P_int * (1 - P_int)
    elif act == "Leaky ReLU":
        P_int = LeakyReLU(np.dot(W_int, s))
        p_out = LeakyReLU(P_int.dot(W_out))
        grad_out = 0.01 if p_out < 0 else 1
        grad_int = [1 if P_int[i] >= 0 else 0.01 for i in range(len(P_int))]
    elif act == "SWISH":
        P_int = SWISH(np.dot(W_int, s))
        p_out = SWISH(P_int.dot(W_out))
        grad_out = SWISH(p_out) + (sigmoid(p_out)*(1 - SWISH(p_out)))
        grad_int = SWISH(P_int) + (sigmoid(P_int)*(1 - SWISH(P_int)))
    elif act == "Fonction tangente hyperbolique":
        P_int = FctTanHyp(np.dot(W_int, s))
        p_out = FctTanHyp(P_int.dot(W_out))
        grad_out = 1 - FctTanHyp(p_out)**2
        grad_int = 1 - FctTanHyp(P_int)**2
    ################################################################

    Delta_int = grad_out * W_out * grad_int
    if learning_strategy[0] == 'Q-learning':
        alpha = learning_strategy[1]
        W_int -= alpha * delta * np.outer(Delta_int, s)
        W_out -= alpha * delta * grad_out * P_int
    elif learning_strategy[0] == 'TD-lambda' or learning_strategy[0] == 'Q-lambda': # Ajout de la stratégie Q-lambda
        alpha = learning_strategy[1]
        lamb = learning_strategy[2]
        Z_int = learning_strategy[3]
        Z_out = learning_strategy[4]
        Z_int *= lamb
        Z_int += np.outer(Delta_int, s)
        # remarque : si au lieu des deux operations ci-dessus nous avions fait
        #    Z_int = lamb*Z_int + np.outer(Delta_int,s)
        # alors cela aura créé un nouveau numpy array, en particulier learning_strategy[3] n'aurait pas été modifié
        Z_out *= lamb
        Z_out += grad_out * P_int
        W_int -= alpha * delta * Z_int
        W_out -= alpha * delta * Z_out