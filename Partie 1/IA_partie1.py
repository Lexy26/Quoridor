from random import randint, random

s=0
p=1

def makeMove(M, last, strategy, eps, alpha):

    psMax = max( [M[i][p] for i in range(len(M))] )

    # greedy ?
    r = random()            # return a float between 0 and 1
    greedy = (r >= eps)

    # choice of movement:
    # greedy case, exploitation: choose a move with psMax
    if greedy:
        moves = [ M[i] for i in range(len(M)) if M[i][p] == psMax ]
        move = moves[randint(0,len(moves)-1)]

    # exploration: choose a random move
    else:
        move = M[randint(0,len(M)-1)]

    # online strategies: update last if not first move:
    if last != None:
        
        if strategy=="TD(0)":
            last[p] = (1-alpha)*last[p] + alpha*move[p]

        elif strategy=="Q-learning":
            last[p] = (1-alpha)*last[p] + alpha*psMax
    
    return move[s]


def endGame(won, history, strategy, alpha):

    # online strategies: update last state
    if strategy == "TD(0)" or strategy == "Q-learning":
        history[-1][p] = (1-alpha) * history[-1][p] + alpha * won

    # offline strategy: update history of states
    elif strategy == "Monte Carlo":
        for i in range(1,len(history)+1):
            history[-i][p] = (1-alpha**i) * history[-i][p] + alpha**i * won
