import numpy as np
import matplotlib.pyplot as plt
import sys

class path_optimization:
    def __init__(self, cost):
        '''
        compute cost surface (numpy array)
        '''
        self.cost = cost
        self.tc = np.zeros_like(cost)
    def cs_origin(self, upward_allowed):
        #helper function to compute cost map (assuming starting at (0,0))
        #origin is defined to be top left corner
        self.tc[0][0] = self.cost[0][0]
        #initial first column
        for i in range(1, self.cost.shape[0] - 1 + 1):
            self.tc[i][0] = self.tc[i-1][0] + self.cost[i][0]
        if upward_allowed:
            #initial first row
            for j in range(1, self.cost.shape[1] - 1 + 1):
                for i in range(self.cost.shape[0] - 1 + 1):
                    if i == 0:
                        self.tc[i][j] = min(self.tc[i][j-1],self.tc[i+1][j-1]) + self.cost[i][j]
                    elif (i != (self.cost.shape[0] - 1)) and (i != 0):
                        self.tc[i][j] = min(self.tc[i-1][j-1], self.tc[i-1][j], self.tc[i][j-1], self.tc[i+1][j-1]) + self.cost[i][j]
                    elif (i == self.cost.shape[0] - 1):
                        self.tc[i][j] = min(self.tc[i-1][j-1], self.tc[i-1][j], self.tc[i][j-1]) + self.cost[i][j]
        else:
            #initial first row
            for j in range(1, self.cost.shape[1] - 1 + 1):
                self.tc[0][j] = self.tc[0][j-1] + self.cost[0][j]
            for i in range(1, self.cost.shape[0] - 1 + 1):
                for j in range(1, self.cost.shape[1] - 1 + 1):
                    self.tc[i][j] = min(self.tc[i-1][j-1], self.tc[i-1][j], self.tc[i][j-1]) + self.cost[i][j]
        return self.tc

    def extract_min_path_origin(self, upward_allowed):
        path=[(0,0)]
        if upward_allowed:
            iter_stats=True
            while iter_stats:
                if (path[-1][0] < self.tc.shape[0]) and (path[-1][0] > 0) and (path[-1][0] != self.tc.shape[0]-1):
                    action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1], #horizontal
                                        self.tc[path[-1][0]+1][path[-1][1]+1], #downward 45
                                        self.tc[path[-1][0]-1][path[-1][1]+1], #upward 45
                                        self.tc[path[-1][0]+1][path[-1][1]]]) #downward vertical
                elif (path[-1][0] < self.tc.shape[0]) and (path[-1][0] == 0):
                    action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1],
                                        self.tc[path[-1][0]+1][path[-1][1]+1],
                                        sys.maxsize,
                                        self.tc[path[-1][0]+1][path[-1][1]]])
                elif (path[-1][0] == self.tc.shape[0]-1):
                    action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1],
                                        sys.maxsize,
                                        self.tc[path[-1][0]-1][path[-1][1]+1],
                                        sys.maxsize])
                if action == 0:
                    path.append((path[-1][0], path[-1][1]+1))
                elif action == 1:
                    path.append((path[-1][0]+1, path[-1][1]+1))
                elif action == 2:
                    path.append((path[-1][0]-1, path[-1][1]+1))
                elif action == 3:
                    path.append((path[-1][0]+1, path[-1][1]))
                if path[-1][0] < 0:
                    raise Exception('Y-Coordinate cannot be less than 0')
                if path[-1][1] == self.tc.shape[1]-1:
                    break
        else:
            iter_stats=True
            while iter_stats:
                if (path[-1][0] < self.tc.shape[0]) and (path[-1][0] > 0) and (path[-1][0] != self.tc.shape[0]-1):
                        action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1], #horizontal
                                            self.tc[path[-1][0]+1][path[-1][1]+1], #downward 45
                                            sys.maxsize, #upward 45
                                            self.tc[path[-1][0]+1][path[-1][1]]]) #downward vertical
                elif (path[-1][0] < self.tc.shape[0]) and (path[-1][0] == 0):
                    action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1],
                                        self.tc[path[-1][0]+1][path[-1][1]+1],
                                        sys.maxsize,
                                        self.tc[path[-1][0]+1][path[-1][1]]])
                elif (path[-1][0] == self.tc.shape[0]-1):
                    action = np.argmin([self.tc[path[-1][0]][path[-1][1]+1],
                                        sys.maxsize,
                                        sys.maxsize,
                                        sys.maxsize])
                if action == 0:
                    path.append((path[-1][0], path[-1][1]+1))
                elif action == 1:
                    path.append((path[-1][0]+1, path[-1][1]+1))
                elif action == 2:
                    path.append((path[-1][0]-1, path[-1][1]+1))
                elif action == 3:
                    path.append((path[-1][0]+1, path[-1][1]))
                if path[-1][0] < 0:
                    raise Exception('Y-Coordinate cannot be less than 0')
                if path[-1][1] == self.tc.shape[1]-1:
                    break
        return path
    
    #dynamic programming optimization
    def min_cost_path_a(self, cost_surface, st_m, st_n, m, n, path=[]):
        '''
        m refers to row selection and n refers to column selection.
        when plotting, it should be reversed
        left to right implementation
        starting point is higher than ending point
        '''
        if m < st_m or n < st_n:
            return sys.maxsize, []
        elif (n == st_n and m == st_m):
            return cost_surface[m][n], [(m,n)]
        else:
            cost_diag, path_diag = self.min_cost_path_a(cost_surface, st_m, st_n , m-1, n-1, path)
            cost_up, path_up = self.min_cost_path_a(cost_surface, st_m, st_n , m-1, n, path)
            cost_left, path_left =  self.min_cost_path_a(cost_surface, st_m, st_n , m, n-1, path)
            min_cost = cost_surface[m][n] + min(cost_diag, cost_up, cost_left)
            if min_cost == cost_surface[m][n] + cost_diag:
                return min_cost, [(m,n)] + path_diag
            elif min_cost == cost_surface[m][n] + cost_up:
                return min_cost, [(m,n)] + path_up
            elif min_cost == cost_surface[m][n] + cost_left:
                return min_cost, [(m,n)] + path_left

    def min_cost_path_b(self, cost_surface, st_m, st_n, m, n, path):
        '''
        m refers to row selection and n refers to column selection.
        when plotting, it should be reversed
        left to right implementation
        starting point is lower than ending point
        '''
        if m > st_m or n < st_n:
            return sys.maxsize, []
        elif (n == st_n and m == st_m):
            return cost_surface[m][n], [(m,n)]
        else:
            cost_diag, path_diag = self.min_cost_path_b(cost_surface, st_m, st_n , m+1, n-1, path)
            cost_down, path_down = self.min_cost_path_b(cost_surface, st_m, st_n , m+1, n, path)
            cost_left, path_left = self.min_cost_path_b(cost_surface, st_m, st_n , m, n-1, path)
            min_cost = cost_surface[m][n] + min(cost_diag, cost_down, cost_left)
            if min_cost == cost_surface[m][n] + cost_diag:
                return min_cost, [(m,n)] + path_diag
            elif min_cost == cost_surface[m][n] + cost_down:
                return min_cost, [(m,n)] + path_down
            elif min_cost == cost_surface[m][n] + cost_left:
                return min_cost, [(m,n)] + path_left
    
class control_path:
    def __init__(self, cost, num_action):
        '''
        compute cost surface (numpy array)
        '''
        self.cost = cost
        self.num_states = len(cost.flatten()) + cost.shape[0]
        self.num_action = num_action
        
    def env_create(self, non_des_reward, target_state):
        '''
        action up = 0, right = 1, down = 2, left = 3
        make non_des_reward several order bigger than cost values
        target_state is the row index where the ending cell is located
        '''
        if type(self.cost) != np.ndarray:
            raise Exception("Cost must be a 2-d numpy array")
        # position of the row in the matrix where the target cell is located
        # always assume that the right-most cells are the terminal state
        if type(target_state) != int:
            raise Exception("Target state has to be integer")
        if non_des_reward > 0 :
            raise Exception("Reward for non-target must be negative")
        # process cost_surface
        cost = -self.cost
        cost[:,-1] = [non_des_reward] * cost.shape[0]
        cost[target_state, cost.shape[1]-1] = -non_des_reward
        # assume entire left column to be terminal (only 1 pixel target)
        # can move anywhere (4 direction)
        # can be extended to include all direction
        # order of values: probability (here is 1 since its deterministic), next state, immediate reward, terminal: boolean
        out_state = list(range(len(self.cost.flatten()), len(self.cost.flatten()) + self.cost.shape[0]))
        env_p = []
        # handle top cell
        for i in range(cost.shape[1]):
            if i == 0:
                env_p.extend([[[[1, i, 0, False]], [[1, i + 1, cost[0][i + 1], False]],
                             [[1, i + cost.shape[1], cost[1][i], False]], [[1, i, 0, False]]]])
            elif i == (cost.shape[1] - 1):
                # terminal states numbering starts from top
                env_p.extend([[[[1, out_state[0], non_des_reward, True]], [[1, out_state[0], non_des_reward, True]], 
                              [[1, out_state[0], non_des_reward, True]], [[1, out_state[0], non_des_reward, True]]]])
            else:
                env_p.extend([[[[1, i, 0, False]], [[1, i + 1, cost[0][i + 1], False]],
                             [[1, i + cost.shape[1], cost[1][i], False]], [[1, i - 1, cost[0][i - 1], False]]]])
        for j in range(1, cost.shape[0]):
            if j != cost.shape[0] - 1:
                env_p.extend([[[[1, (j-1) * cost.shape[1], cost[j-1][0], False]], [[1, j * cost.shape[1] + 1, cost[j][1], False]],
                                 [[1, (j+1) * cost.shape[1], cost[j+1][0], False]], [[1, j * cost.shape[1], 0, False]]]])
            # handle bottom cell
            elif j == cost.shape[0] - 1:
                env_p.extend([[[[1, (j-1) * cost.shape[1], cost[j-1][0], False]], [[1, j * cost.shape[1] + 1, cost[j][1], False]],
                                 [[1, j * cost.shape[1], 0, False]], [[1, j * cost.shape[1], 0, False]]]])
            for k in range(1, cost.shape[1]):
                if k != cost.shape[1] - 1 and j != cost.shape[0] - 1:
                    env_p.extend([[[[1, (j-1) * cost.shape[1] + k, cost[j-1][k], False]], 
                                   [[1, j * cost.shape[1] + k + 1, cost[j][k+1], False]], 
                                   [[1, (j+1) * cost.shape[1] + k, cost[j+1][k], False]], 
                                   [[1, j * cost.shape[1] + k - 1, cost[j][k-1], False]]]])
                if k != cost.shape[1] - 1 and j == cost.shape[0] - 1:
                    env_p.extend([[[[1, (j-1) * cost.shape[1] + k, cost[j-1][k], False]], 
                                   [[1, j * cost.shape[1] + k + 1, cost[j][k+1], False]], 
                                   [[1, j * cost.shape[1] + k, 0, False]], 
                                   [[1, j * cost.shape[1] + k - 1, cost[j][k-1], False]]]])
                elif k == cost.shape[1] - 1 and j != cost.shape[0] - 1:
                    env_p.extend([[[[1, out_state[j], non_des_reward, True]],
                                   [[1, out_state[j], non_des_reward, True]],
                                   [[1, out_state[j], non_des_reward, True]], 
                                   [[1, out_state[j], non_des_reward, True]]]])
                elif k == cost.shape[1] - 1 and j == cost.shape[0] - 1:
                    env_p.extend([[[[1, out_state[j], non_des_reward, True]],
                                   [[1, out_state[j], non_des_reward, True]],
                                   [[1, out_state[j], non_des_reward, True]], 
                                   [[1, out_state[j], non_des_reward, True]]]])
        for z in range(len(out_state)):
            env_p.extend([[[[1, out_state[z], 0, True]], [[1, out_state[z], 0, True]], 
                           [[1, out_state[z], 0, True]], [[1, out_state[z], 0, True]]]])
        # replace entry to target state   
        env_p[(target_state+1) * cost.shape[1] - 1] = [[[1, out_state[target_state], -non_des_reward, True]],
                                                       [[1, out_state[target_state], -non_des_reward, True]],
                                                       [[1, out_state[target_state], -non_des_reward, True]],
                                                       [[1, out_state[target_state], -non_des_reward, True]]]
        return env_p
    
    def value_iter(self, n_iter, env_p):
        ns = n_iter
        self.state = list(range(self.num_states))
        action = [0,1,2,3] # north=north, north=east, north=south, north=west
        v = np.zeros((ns,len(self.state)))
        # iterate over the cells and its value
        for i in range(1,ns):
            for each_state in self.state:
                a = np.zeros(len(action))
                for each_action in action:
                    for prob, next_state, reward, done in env_p[each_state][each_action]:
                        a[each_action] += prob * (reward + 0.9 * v[i-1,next_state])
                v[i,each_state]=np.round(np.max(a),2)
        return v
    
    def one_step_lookahead(self, state, env_p, v):
        '''
        1. call env_create to generate the third argument env_p
        2. call value_iter to generate the updated cell values
        '''
        # helper function
        a = np.zeros(self.num_action)
        for i in range(self.num_action):
            for prob, next_state, reward, done in env_p[state][i]:
                a[i] += prob * (reward + 0.9 * v[next_state])
        return a
    
    def greedy_policy(self, state, env_p, v):
        val = self.one_step_lookahead(state, env_p, v)
        return np.argmax(val)
    
    def extract_policy(self, env_p, v):
        policy_pi = []
        for i in range(len(self.state)):
            policy_pi.append(self.greedy_policy(i, env_p, v))
        policy_dict = dict(zip(range(self.num_states), policy_pi))
        return policy_dict
    
    def state_to_cord(self):
        '''
        helper function to map state label and coordinates
        '''
        if type(self.cost) != np.ndarray:
            raise Exception("Cost must be a numpy array")
        cord = []
        for i in range(self.cost.shape[0]):
            for j in range(self.cost.shape[1]):
                cord.append((i,j))
        self.cord_dict = dict(zip(range(self.num_states), cord))
        return self.cord_dict
    
    def policy_path(self, policy, start_state, end_state):
        '''
        Policy is a dictionary
        State is the flattened ordered 0 to len(num_states)
        End state is the actual state label
        '''
        cord = self.state_to_cord()
        st_m, st_n = cord[start_state]
        m, n = cord[end_state]
        path = [(st_m, st_n)]
        path_state = [start_state]
        i = 0
        while i < sys.maxsize:
            action = policy[path_state[-1]]
            if action == 0:
                path.append((path[-1][0] - 1, path[-1][1]))
                path_state.append(path_state[-1] - self.cost.shape[1])
            elif action == 1:
                path.append((path[-1][0], path[-1][1] + 1))
                path_state.append(path_state[-1] + 1)
            elif action == 2:
                path.append((path[-1][0] + 1, path[-1][1]))
                path_state.append(path_state[-1] + self.cost.shape[1])
            else:
                path.append((path[-1][0], path[-1][1] - 1))
                path_state.append(path_state[-1] - 1)
            if path[-1][0] == m and path[-1][1] == n:
                break
            i += 1
        return path
