"""
    Auteur : Alessia Stefanelli - 464273
    Date : 07/04/2019
    Objet : Partie 4
"""

import time, random
import numpy as np
import random
import os, sys
from select import select
""" Ce fichier n'a pas été modifié """

def clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')


def waitForKey():
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK
                  | termios.ISTRIP | termios.INLCR | termios.IGNCR
                  | termios.ICRNL | termios.IXON)
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios.PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
        if ret == '\x1b':
            ret += sys.stdin.read(2)
    except KeyboardInterrupt:
        ret = '\x03'
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


def wait(timeout):
    rlist, wlist, xlist = select([sys.stdin], [], [], timeout)


def progressBar(i, n):
    return '  ' + str(int(100 * i / n)) + '%'


def listEncoding(board, N, WALLS):
    # outputs list encoding of board:
    # [ [i, j], [k, l], list_of_horizontal_walls, list_of_vertical_walls, walls_left_p1, walls_left_p2 ]
    # where [i, j] position of white player and [k, l] position of black player
    # and each wall in lists of walls is of the form [a, b] where [a,b] is the south-west square
    pos = [None,None]
    coord = [ [None,None], [None,None]]
    walls = [ [], [] ]
    walls_left = [ None, None ]
    for i in range(2):
        pos[i] = board[i*N**2:(i+1)*N**2].argmax()
        coord[i][0] = pos[i]%N
        coord[i][1] = pos[i]//N
        for j in range((N-1)**2):
            if board[2*N**2 + i*(N-1)**2 + j]==1:
                walls[i].append( [j%(N-1), j//(N-1)] )
        walls_left[i] = board[2*N**2 + 2*(N-1)**2 + i*(WALLS+1):2*N**2 + 2*(N-1)**2 + (i+1)*(WALLS+1)].argmax()
    return [ coord[0], coord[1], walls[0], walls[1], walls_left[0], walls_left[1] ]


def canMove(board, coord, step, N, WALLS):
    # returns True if there is no wall in direction step from pos, and we stay in the board
    # NB: it does not check whether the destination is occupied by a player
    new_coord = coord + step
    in_board = new_coord.min() >= 0 and new_coord.max() <= N - 1
    if not in_board:
        return False
    if WALLS > 0:
        if step[0] == -1:
            L = []
            if new_coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (new_coord[1] - 1) * (N - 1) + new_coord[0])
        elif step[0] == 1:
            L = []
            if coord[1] < N - 1:
                L.append(2 * N ** 2 + (N - 1) ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[1] > 0:
                L.append(2 * N ** 2 + (N - 1) ** 2 + (coord[1] - 1) * (N - 1) + coord[0])
        elif step[1] == -1:
            L = []
            if new_coord[0] < N - 1:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0])
            if new_coord[0] > 0:
                L.append(2 * N ** 2 + new_coord[1] * (N - 1) + new_coord[0] - 1)
        elif step[1] == 1:
            L = []
            if coord[0] < N - 1:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0])
            if coord[0] > 0:
                L.append(2 * N ** 2 + coord[1] * (N - 1) + coord[0] - 1)
        else:
            print('step vector', step, 'is not valid')
            quit(1)
        if sum([board[j] for j in L]) > 0:
            # move blocked by a wall
            return False
    return True

def listMoves(board, current_player, N, WALLS,G):
    if current_player not in [0, 1]:
        print('error in function listMoves: current_player =', current_player)
    pn = current_player
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    moves = []
    pos = [None, None]
    coord = [None, None]
    for i in range(2):
        pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
        coord[i] = np.array([pos[i] % N, pos[i] // N])
        pos[i] += pn * N ** 2  # offset for black player
    P = []  # list of new boards (each encoded as list bits to switch)
    # current player moves to another position
    for s in steps:
        if canMove(board, coord[pn], s, N, WALLS):
            new_coord = coord[pn] + s
            new_pos = pos[pn] + s[0] + N * s[1]
            occupied = np.array_equal(new_coord, coord[(pn + 1) % 2])
            if not occupied:
                P.append([pos[pn], new_pos])  # new board is obtained by switching these two bits
            else:
                can_jump_straight = canMove(board, new_coord, s, N, WALLS)
                if can_jump_straight:
                    new_pos = new_pos + s[0] + N * s[1]
                    P.append([pos[pn], new_pos])
                else:
                    if s[0] == 0:
                        D = [(-1, 0), (1, 0)]
                    else:
                        D = [(0, -1), (0, 1)]
                    for i in range(len(D)):
                        D[i] = np.array(D[i])
                    for d in D:
                        if canMove(board, new_coord, d, N, WALLS):
                            final_pos = new_pos + d[0] + N * d[1]
                            P.append([pos[pn], final_pos])
                            # current player puts down a wall
    # TO DO: Speed up this part: it would perhaps be faster to directly discard intersecting walls based on existing ones
    nb_walls_left = board[
                    2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1):2 * N ** 2 + 2 * (N - 1) ** 2 + (pn + 1) * (
                                WALLS + 1)].argmax()
    ind_walls_left = 2 * N ** 2 + 2 * (N - 1) ** 2 + pn * (WALLS + 1) + nb_walls_left
    if nb_walls_left > 0:
        for i in range(2 * (N - 1) ** 2):
            pos = 2 * N ** 2 + i
            L = [pos]  # indices of walls that could intersect
            if i < (N - 1) ** 2:
                # horizontal wall
                L.append(pos + (N - 1) ** 2)  # vertical wall on the same 4-square
                if i % (N - 1) > 0:
                    L.append(pos - 1)
                if i % (N - 1) < N - 2:
                    L.append(pos + 1)
            else:
                # vertical wall
                L.append(pos - (N - 1) ** 2)  # horizontal wall on the same 4-square
                if (i - (N - 1) ** 2) // (N - 1) > 0:
                    L.append(pos - (N - 1))
                if (i - (N - 1) ** 2) // (N - 1) < N - 2:
                    L.append(pos + (N - 1))
            nb_intersecting_wall = sum([board[j] for j in L])
            if nb_intersecting_wall == 0:
                board[pos] = 1
                # we remove the corresponding edges from G
                if i < (N - 1) ** 2:
                    # horizontal wall
                    a, b = i % (N - 1), i // (N - 1)
                    E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                else:
                    # vertical wall
                    a, b = (i - (N - 1) ** 2) % (N - 1), (i - (N - 1) ** 2) // (N - 1)
                    E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
                if eachPlayerHasPath(board, N, G):
                    P.append(
                        [pos, ind_walls_left - 1, ind_walls_left])  # put down the wall and adapt player's counter
                board[pos] = 0
                # we add back the two edges in G
                for e in E:
                    G[e[0]][e[1]][e[2]] = 1
                    # we create the new boards from P
    for L in P:
        new_board = board.copy()
        for i in L:
            new_board[i] = not new_board[i]
        moves.append(new_board)

    return moves

def endOfGame(board,N):
    return board[(N - 1) * N:N ** 2].max() == 1 or board[N ** 2:N ** 2 + N].max() == 1

def startingBoard(N, WALLS):
    board = np.array([0] * (2 * N ** 2 + 2 * (N - 1) ** 2 + 2 * (WALLS + 1)))
    # player positions
    board[(N - 1) // 2] = True
    board[N ** 2 + N * (N - 1) + (N - 1) // 2] = True
    # wall counts
    for i in range(2):
        board[2 * N ** 2 + 2 * (N - 1) ** 2 + i * (WALLS + 1) + WALLS] = 1
    return board


def eachPlayerHasPath(board, N, G):
    # heuristic when at most one wall
    nb_walls = board[2 * N ** 2:2 * N ** 2 + 2 * (N - 1) ** 2].sum()
    if nb_walls <= 1:
        # there is always a path when there is at most one wall
        return True
    # checks whether the two players can each go to the opposite side
    pos = [None,None]
    coord = [[None, None], [None, None]]
    for i in range(2):
        pos[i] = board[i * N ** 2:(i + 1) * N ** 2].argmax()
        coord[i][0] = pos[i] % N
        coord[i][1] = pos[i] // N
        coord[i] = np.array(coord[i])
    steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i in range(len(steps)):
        steps[i] = np.array(steps[i])
    for i in range(2):
        A = np.zeros((N, N), dtype='bool')  # TO DO: this could be optimized
        S = [coord[i]]  # set of nodes left to treat
        finished = False
        while len(S) > 0 and not finished:
            c = S.pop()
            # NB: In A we swap rows and columns for simplicity
            A[c[1]][c[0]] = True
            for k in range(4):
                if G[c[0]][c[1]][k] == 1:
                    s = steps[k]
                    new_c = c + s
                    # test whether we reached the opposite row
                    if i == 0:
                        if new_c[1] == N - 1:
                            finished = True
                            break
                    else:
                        if new_c[1] == 0:
                            return True
                    # otherwise we continue exploring
                    if A[new_c[1]][new_c[0]] == False:
                        # heuristic, we give priority to moves going up (down) in case player is white (black)
                        if i == 0:
                            if k == 1:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
                        else:
                            if k == 3:
                                S.append(new_c)
                            else:
                                S.insert(0, new_c)
        if not finished:
            return False
    return True


def sigmoid(x):
    """Cette fonction calcule la valeur de la fonction sigmoide (de 0 à 1) pour un nombre réel donné.
    Il est à noter que x peut également être un vecteur de type numpy array, dans ce cas, la valeur de la sigmoide correspondante à chaque réel du vecteur est calculée.
    En retour de la fonction, il peut donc y avoir un objet de type numpy.float64 ou un numpy array."""
    return 1 / (1 + np.exp(-x))


def SWISH(x):
    return x*sigmoid(x)


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
    P_int = SWISH(np.dot(W_int, s))
    p_out = SWISH(P_int.dot(W_out))
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
    P_int = SWISH(np.dot(W_int, s))
    p_out = SWISH(P_int.dot(W_out))
    grad_out = SWISH(p_out) + (sigmoid(p_out)*(1 - SWISH(p_out)))
    grad_int = SWISH(P_int) + (sigmoid(P_int)*(1 - SWISH(P_int)))
    Delta_int = grad_out * W_out * grad_int
    alpha = learning_strategy[1]
    W_int -= alpha * delta * np.outer(Delta_int, s)
    W_out -= alpha * delta * grad_out * P_int


class Player_AI():
    def __init__(self, act, NN, eps, learning_strategy, name='IA', G=None, N=5, WALLS=3):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0
        self.NN = NN
        self.eps = 0.60
        self.learning_strategy = learning_strategy
        self.G = G
        self.N = N
        self.WALLS = WALLS
        self.ACT = act  # Ajout de l'attribut pour la fonction d'Activation


    def makeMove(self, board):
        return self.makeThinkingMove(listMoves(board, self.color, self.N, self.WALLS, self.G), board, self.color,
                                     self.NN, self.eps, self.learning_strategy, self.ACT)

    def makeThinkingMove(self, moves, s, color, NN, eps, act, learning_strategy='Q-learning'):
        """Fonction appelée pour que l'IA choisisse une action (le nouvel état new_s est retourné)
             à partir d'une liste d'actions possibles "moves" (c'est-à-dire une liste possible d'états accessibles) à partir d'un état actuel (s)
             Un réseau de neurones (NN) est nécessaire pour faire ce choix quand le mouvement n'est pas du hasard.
             Dans le cas greedy (non aléatoire), la couleur du joueur dont c'est le tour sera utilisée pour déterminer s'il faut retenir le meilleur ou le pire coup du point de vue du joueur blanc.
             Le cas greedy survient avec une probabilité 1-eps.
             La stratégie d'apprentissage fait référence à la stratégie utilisée dans la fonction de backpropagation (cfr la fonction de backpropagation pour une description)
             """
        Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
        # Epsilon greedy
        greedy = random.random() > eps

        # dans le cas greedy, on recherche le meilleur mouvement (état) possible. Dans le cas du Q-learning (même sans greedy), on a besoin de connaître
        # la probabilité estimée associée au meilleur mouvement (état) possible en vue de réaliser la backpropagation.
        if greedy or Q_learning:
            best_moves = []
            best_value = None
            c = 1
            if color == 1:
                # au cas où c'est noir qui joue, on s'interessera aux pires coups du point de vue de blanc
                c = -1
            for m in moves:
                val = forwardPass(m, NN, act)
                if best_value == None or c * val > c * best_value:  # si noir joue, c'est comme si on regarde alors si val < best_value
                    best_moves = [m]
                    best_value = val
                elif val == best_value:
                    best_moves.append(m)

        if greedy:
            # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
            new_s = best_moves[random.randint(0, len(best_moves) - 1)]
        else:
            # on choisit un mouvement au hasard
            new_s = moves[random.randint(0, len(moves) - 1)]

            # on met à jour les poids si nécessaire
        if Q_learning:
            p_out_s = forwardPass(s, NN, act)
            delta = p_out_s - best_value
            backpropagation(act, s, NN, delta, learning_strategy)

        return new_s

    def CalculateEndGame(self, s, won, NN, learning_strategy, act):
        Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
        # on met à jour les poids si nécessaire
        if Q_learning:
            p_out_s = forwardPass(s, NN, act)
            delta = p_out_s - won
            backpropagation(act, s, NN, delta, learning_strategy)
    def endGame(self, board, won):
        self.CalculateEndGame(board, won, self.NN, self.learning_strategy, self.ACT)


class Grid_Board():
    """
    Game class with all parameters
    """
    def __init__(self, N=5, Walls=1, eps=0.35, alpha=0.15, lamb=0.9, ls='Q-learning',
                 G=None, G_init=None, NN=70, act='SWISH', e_greedy='Normal'):
        super(Grid_Board, self).__init__()
        self.N = N  # N*N board
        self.WALLS = Walls  # number of walls each player has
        self.EPS = eps  # epsilon in epsilon-greedy
        self.ALPHA = alpha  # learning_rate
        self.LAMB = lamb  # lambda for TD(lambda)
        self.LS = ls  # default learning strategy
        self.G = G  # graph of board (used for connectivity checks)
        self.G_INIT = self.computeGraph()
        self.NN = createNN(2 * self.N ** 2 + 2 * (self.N - 1) ** 2 + 2 * (self.WALLS + 1), NN)
        self.current_player = None
        self.ACT = act  # Ajout de l'attribut pour la fonction d'Activation
        self.E = e_greedy  # Ajout de l'attribut pour l'e-Greedy

    def save(self, name):
        """
        Save the AI in a given filename
        """
        filename = name
        np.savez(filename, N=self.N, WALLS=self.WALLS, W1=self.NN[0], W2=self.NN[1])


    def playGame(self, player1, player2, show=False, delay=0.0):
        # initialization
        players = [player1, player2]
        board = startingBoard(self.N, self.WALLS)
        self.G = self.G_INIT.copy()
        if show:
            self.moved.emit(listEncoding(board, self.N, self.WALLS), ['', MSG_MOVEMENT])
        for i in range(2):
            players[i].color = i
        # main loop
        finished = False
        current_player = 0
        count = 0
        quit = False
        msg = ''
        while not finished:
            if show:
                msg = ''
                txt = ['Blanc', 'Noir ']
                for i in range(2):
                    if i == current_player:
                        msg += '* '
                    else:
                        msg += '  '
                    msg += txt[i] + ' : ' + players[i].name
                    msg += '\n'
                for i in range(2):
                    if 'IA' in players[i].name:
                        # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                        p = forwardPass(board, players[i].NN, self.ACT)
                        msg += '\nEstimation IA : ' + "{0:.4f}".format(p)
                        msg += '\n'
                self.moved.emit(listEncoding(board, self.N, self.WALLS), [msg, MSG_MOVEMENT])
                self.moved.emit(listEncoding(board, self.N, self.WALLS), [msg, MSG_MOVEMENT])
                time.sleep(0.3)
            self.current_player = players[current_player]
            new_board = self.current_player.makeMove(board) if 'IA' in self.current_player.name else self.makeMove(
                self.current_player, board)
            # we compute changes of G (if any) to avoid recomputing G at beginning of listMoves
            # we remove the corresponding edges from G
            if not new_board is None:
                v = new_board[2 * self.N ** 2:2 * self.N ** 2 + 2 * (self.N - 1) ** 2] - board[
                                                                                         2 * self.N ** 2:2 * self.N ** 2 + 2 * (
                                                                                                 self.N - 1) ** 2]
                i = v.argmax()
                if v[i] == 1:
                    # a wall has been added, we remove the two corresponding edges of G
                    if i < (self.N - 1) ** 2:
                        # horizontal wall
                        a, b = i % (self.N - 1), i // (self.N - 1)
                        E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                    else:
                        # vertical wall
                        a, b = (i - (self.N - 1) ** 2) % (self.N - 1), (i - (self.N - 1) ** 2) // (self.N - 1)
                        E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                    for e in E:
                        self.G[e[0]][e[1]][e[2]] = 0
            board = new_board

            if show:
                self.moved.emit(listEncoding(board, self.N, self.WALLS),
                                [msg, MSG_MOVEMENT]) if board is not None else None
            if board is None:
                # human player quit
                quit = True
                finished = True
                if show:
                    self.exit.emit(['', END_GAME])
            elif endOfGame(board, self.N):
                players[current_player].score += 1
                white_won = current_player == 0
                players[current_player].endGame(board, white_won)
                if show:
                    self.moved.emit(listEncoding(board, self.N, self.WALLS), ['', END_GAME])
                    time.sleep(0.1)
                finished = True
            else:
                current_player = (current_player + 1) % 2
        return quit

    def computeGraph(self, board=None):
        # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
        pos_steps = [(1, 0), (0, 1)]
        for i in range(len(pos_steps)):
            pos_steps[i] = np.array(pos_steps[i])
        g = np.zeros((self.N, self.N, 4))
        for i in range(self.N):
            for j in range(self.N):
                c = np.array([i, j])
                for k in range(2):
                    s = pos_steps[k]
                    if board is None:
                        # initial setup
                        new_c = c + s
                        if new_c.min() >= 0 and new_c.max() <= self.N - 1:
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
                    else:
                        if canMove(board, c, s, self.N, self.WALLS):
                            new_c = c + s
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
        return g

    def train(self, n_train=10000):
        """
        Train the IA with current parameters
        :param NN: neural network created according to the parameters of the GUI
        :param n_train: number of turns to train the IA
        """
        NN = self.NN
        if self.LS == 'Q-learning':
            learning_strategy1 = (self.LS, self.ALPHA)
            learning_strategy2 = (self.LS, self.ALPHA)
        agent1 = Player_AI(self.ACT, NN, self.EPS, learning_strategy1, 'IA 1', G=self.G_INIT, N=self.N,
                           WALLS=self.WALLS)
        agent2 = Player_AI(self.ACT, NN, self.EPS, learning_strategy2, 'IA 2', G=self.G_INIT, N=self.N,
                           WALLS=self.WALLS)
        # training session
        for j in range(n_train):
            self.playGame(agent1, agent2)

def main():
    game = Grid_Board()
    game.train(int(sys.argv[1]))
    game.save(sys.argv[2])
    print('fine')

if __name__ == "__main__":
    main()