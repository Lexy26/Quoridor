import copy, os, time, sys, random, math
import IA_partie1

# Board encoding: 
# [ [i, j], [k, l], list_of_horizontal_walls, list_of_vertical_walls ]
# where [i, j] position of white player and [k, l] position of black player
# and each wall in lists of walls is of the form [a, b] where [a,b] 
# is the north-west square 

N = 5    # N*N board


def clearScreen():
    os.system('cls' if os.name == 'nt' else 'clear')

def waitForKey():
    import termios, fcntl, sys, os
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save) # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK 
                  | termios.ISTRIP | termios.INLCR | termios. IGNCR 
                  | termios.ICRNL | termios.IXON )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON
                  | termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1) # returns a single character
    except KeyboardInterrupt: 
        ret = '\x03'
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


def menu(eps, alpha):
    clearScreen()
    board_list = [ [[(N-1)//2,0],[(N-1)//2,N-1],[], []], [[(N-1)//2,0],[(N-1)//2,N-1],[[1,3], [2,2]], []], [[(N-1)//2,0],[(N-1)//2,N-1],[[0,4], [2,4], [1,3], [3,3], [0,2], [2,2], [1,1], [3,1]], []] ]
    board_menu = [ ]
    for i in range(len(board_list)):
        board_menu.append(str(i+1))    
    IA_menu = ['Aléatoire', 'Monte Carlo', 'TD(0)', 'Q-learning']
    white_menu = IA_menu[:]
    black_menu = IA_menu[:]
    iter_menu =  [ 128 ]    
    for i in range(10): 
        iter_menu.append(iter_menu[-1]*2)    
    whole_menu = [ board_menu, white_menu, black_menu, iter_menu ]
    selected = [ 0 ]*len(whole_menu)    
    menu_titles = ['Plateau', 'Joueur blanc', 'Joueur noir', 'Itérations']
    maxspace_menu = [ len(title) for title in menu_titles]
    for i in range(len(whole_menu)):
        maxspace_menu[i] = max([len(str(elem)) for elem in whole_menu[i]] + [maxspace_menu[i]])
    current_menu = 0
    UP = 'i'
    DOWN = 'k'
    LEFT = 'j'
    RIGHT = 'l'    
    while True:                
        displayBoard(board_list[selected[0]])
        h_offset = '' 
        offset_in_between = ' '*5        
        s = h_offset + 'eps = ' + str(eps) + '\n' + h_offset + 'alpha = ' + str(alpha) + '\n'
        print(s)
        s = h_offset + 'sélectionner options avec touches i,j,k,l \n'
        print(s)
        s = h_offset
        for i in range(len(whole_menu)):
            txt = str(menu_titles[i])
            s += txt + ' '*(maxspace_menu[i]-len(txt)) + offset_in_between
        print(s)    
        s = h_offset
        for i in range(len(whole_menu)):
            if current_menu == i:
                txt = '-'*maxspace_menu[i]
            else:
                txt = ''
            s += txt + ' '*(maxspace_menu[i]-len(txt)) + offset_in_between
        print(s)            
        s = h_offset
        for i in range(len(whole_menu)):
            txt = str(whole_menu[i][selected[i]])
            s += txt + ' '*(maxspace_menu[i]-len(txt)) + offset_in_between
        print(s)        
        key = waitForKey()
        if key == UP:
            selected[current_menu] = (selected[current_menu]-1) % len(whole_menu[current_menu])
        elif key == DOWN:
            selected[current_menu] = (selected[current_menu]+1) % len(whole_menu[current_menu])
        elif key == LEFT:
            current_menu = (current_menu-1) % len(whole_menu)
        elif key == RIGHT:
            current_menu = (current_menu+1) % len(whole_menu)
        else:
            res = []
            for i in range(len(whole_menu)):
                if i == 0:
                    res.append(board_list[selected[i]])            
                else:
                    res.append(whole_menu[i][selected[i]])            
            return res



def displayBoard(board, scores=None):     
    clearScreen()
    square_char = '\u25A2'
    #player_char = ('\u25CF','\u25CD')    # for white on black background
    player_char = ('\u25CD','\u25CF')    # for black on white background
    wall_char = ('\u2014', '\uFF5C')
    h_offset = 8
    s = ''
    for j in range(N): 
        line = ' '*h_offset
        hwall = ' '*h_offset
        for i in range(N):
            old_line = line 
            for p in range(2):
                if [i, j] == board[p]:
                    line += player_char[p]
            if line == old_line:
                line += square_char
            beginning_vertical_wall = [i, j] in board[3]
            ending_vertical_wall = j < N-1 and [i, j+1] in board[3]
            if beginning_vertical_wall:                 
                line += ' ' + wall_char[1]  
            elif ending_vertical_wall: 
                line += ' ' + wall_char[1]  
            else:
                line += '   '
            if j > 0: 
                old_hwall = hwall
                for wall in board[2]:            
                    if wall == [i, j]: 
                        hwall += wall_char[0]*4
                    elif wall == [i-1, j]: 
                        hwall += wall_char[0]*2
                        if beginning_vertical_wall:                         
                            hwall += wall_char[1] 
                        else:
                            hwall += ' '*2
                if hwall==old_hwall:
                    if beginning_vertical_wall:                         
                        hwall += '  ' + wall_char[1] 
                    else:
                        hwall += ' '*4
                    
        s =  line + '\n' + hwall + '\n' + s         
    H = [ ' '*(h_offset-2) ]*2
    s = (H[0] + '\n')*2 + '\n' + s + (H[1] + '\n')*2
    if scores == None:
        scores = ''
    print('\n'*2 + s + '\n'*2, scores)        
    
def noWallInDirection(board, pos, step):
    # returns True if there is no wall in direction step from pos  (border of board is treated as a wall)
    [ i, j ] = pos
    new_i = i + step[0]
    new_j = j + step[1]        
    in_board = 0 <= new_i < N and 0 <= new_j < N
    if not in_board:
        return False
    d = abs(step[0])            
    for wall in board[2 + d]:                    
        if d == 0 and wall[0] in [i, i-1]:
            if new_j < wall[1] <= j or j < wall[1] <= new_j:
                return False
        if d == 1 and wall[1] in [j, j+1]:
            if new_i <= wall[0] < i or i <= wall[0] < new_i:
                return False
    return True

def eachPlayerHasPath(board): 
    steps = [ [-1,0], [1,0], [0,-1], [0,1] ]      
    for p in range(2):
        pos = board[p]
        L = [ ]  # set of reachable squares
        has_path = False
        if pos[1] != (N-1)*((p+1)%2):
            A = [ pos ] 
        else: 
            has_path = True
        while len(A)>0:
            [i,j] = A[0]
            A = A[1:]
            L.append([i,j])
            for s in steps:
                new_i = i + s[0]
                new_j = j + s[1]                                        
                if [new_i,new_j] not in L and [new_i,new_j] not in A and noWallInDirection(board, [i,j], s): 
                    if new_j == (N-1)*((p+1)%2):
                        # player p has a path
                        A = []
                        has_path = True
                        break
                    A.append([new_i,new_j])
        if not has_path:
            return False
    return True

def convertToTuples(L):
    # recursively converts the list L into tuples
    if type(L) == type([]):
        T = []
        for item in L: 
            T.append(convertToTuples(item))
        return tuple(T)
    else: 
        return L

def convertToLists(L):
    # recursively converts tuples to lists
    if type(L) == type((1,2)):
        T = []
        for item in L: 
            T.append(convertToLists(item))
        return T
    else: 
        return L


def listMoves(board, current_player):            
    steps = [ [-1,0], [1,0], [0,-1], [0,1] ]      
    moves = []        
    [ i, j ] = board[current_player]    
    P = [] # list of positions where the player can go to
    for s in steps:
        new_i = i + s[0]
        new_j = j + s[1]                
        no_wall = noWallInDirection(board, [i,j], s)
        add = False
        if no_wall:             
            occupied = [new_i, new_j] == board[(current_player+1)%2]                
            if not occupied: 
                P.append([ new_i, new_j])
            else: 
                jump_blocked_by_wall = not noWallInDirection(board, [new_i,new_j], s)
                new_i = new_i + s[0]
                new_j = new_j + s[1]                        
                if jump_blocked_by_wall:
                    new_i = new_i - s[0]
                    new_j = new_j - s[1]                                
                    if s[0]==0:
                        D = [ [-1, 0], [1, 0] ]
                    else:
                        D = [ [0, -1], [0, 1] ]
                    for d in D:
                        if noWallInDirection(board, [new_i,new_j], d): 
                            [ pos_i, pos_j] = [ new_i + d[0], new_j + d[1] ] 
                            P.append([ pos_i, pos_j])
                else:
                    P.append([ new_i, new_j])
    for pos in P: 
        new_board = copy.deepcopy(board)
        new_board[current_player] = pos
        moves.append(new_board)

    return moves



def endOfGame(board):
    return board[0][1]==N-1 or board[1][1]==0

def main_loop(board, players, scores, iter, eps, alpha, show=False):    
    history = [ [], [] ]    
    delay = .15    
    # main loop
    finished = False
    current_player = 0
    while not finished:                 
        if show:
            displayBoard(board, scores)        
            time.sleep(delay)                
        else:
            if iter%10 == 0:
                print('Itérations :', iter, end='\r')
        raw_M = convertToTuples(listMoves(board, current_player))
        M = []
        for state in raw_M:
            if state not in players[current_player][1].keys():
                # this afterstate is encountered for the first time
                players[current_player][1][state] = .5  # no knowledge about probability of winning from this state yet        
            M.append( [ state, players[current_player][1][state] ] )        
        if history[current_player] != []:
            last = history[current_player][-1]
        else:
            last = None
        if players[current_player][0] == 'Aléatoire':
            board_tuple = M[ random.randint(0, len(M)-1) ][0]
        else:
            board_tuple = IA_partie1.makeMove(M, last, players[current_player][0], eps, alpha)
        if players[current_player][0] in ['TD(0)', 'Q-learning'] and history[current_player] != []:
            # probability of the last state in history got updated, we update the dictionary accordingly
            players[current_player][1][history[current_player][-1][0]] = history[current_player][-1][1]
        # we add the new move to the history
        history[current_player].append( [ board_tuple, players[current_player][1][board_tuple] ] )        
        board = convertToLists(board_tuple)
        if endOfGame(board):
            won = [ False, False ]
            won[current_player] = True
            scores[current_player] += 1
            for i in range(2):
                if players[i][0] != 'Aléatoire':
                    IA_partie1.endGame(won[i], history[i], players[i][0], alpha)            
                if players[i][0] in ['TD(0)', 'Q-learning'] and history[i] != []:
                    # probability of the last state in history got updated, we update the dictionary accordingly
                    players[i][1][history[i][-1][0]] = history[i][-1][1]
                elif players[i][0] == 'Monte Carlo':
                    # here we review the whole history
                    for j in range(len(history[i])):
                        players[i][1][history[i][j][0]] = history[i][j][1]
            if show:
                displayBoard(board, scores)                        
                time.sleep(0.3)
            finished = True
        else: 
            current_player = (current_player+1)%2
    return scores

def main():
    global N    
    eps = 0.1
    alpha = 0.3 
    if len(sys.argv) > 1:
        eps = float(sys.argv[1])
        alpha = float(sys.argv[2])
    [ board, white_strategy, black_strategy, nb_iter ] = menu(eps, alpha)    
    scores = [0,0]
    # Markov chains are initially empty
    players = [ [ white_strategy, {} ], [ black_strategy, {} ] ]
    show = False
    print('\nEntraînement')
    for i in range(nb_iter*2):
        if i >= nb_iter:
            show = True        
        scores = main_loop(board, players, scores, i, eps, alpha, show)

            
if __name__ == '__main__':
    main()

