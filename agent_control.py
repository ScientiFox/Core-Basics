###
#EDA project agent control
# A number of agent-control systems
#
# Contents:
#    - Generalized State Machines
#    - Subsumption
#    - State Machine
#    - Q-learning
#    - Murin
#    
###

#Standards
import math,time,random

#For parallelism
import threading

#Specific imports from other core library
import handy_assets
from handy_assets import variable

#For math
import numpy as np

#A standard Q-Learning implementation, but with options for learning rate,
# memory depth, proportional learning, and different action-choice modes
class QlearningFlex:

    #Initialize learner
    def __init__(self,S,A,_alpha,_depth,_Rmax,_init=0):
        self.Q = [] #Quality matrix
        self.Rmax = _Rmax #Maximum reward
        for s in range(S): #Looping over states
            Sa = [] #State/action slice
            Sm = 0.0 #Slice sums
            for a in range(A):
                if _init == 0: #Random initialization
                    Sa = Sa + [random.random()]
                    Sm = Sm + Sa[-1]
                else: #Proportional initialization
                    Sa = Sa + [1.0/A]
                    Sm = Sm + Sa[-1]
            self.Q = self.Q + [(Sa,Sm)] #Update Q array
        self.alpha = _alpha #Set alpha learning rate
        self.depth = _depth #Set memory depth
        self.history = [] #State/action history
        self.S = S #Number of states
        self.A = A #Number of actions

    #Wrapper function to load another Q array
    def loadQ(selfm_Q):
        self.Q = _Q

    #Method to get an action
    def act(self,state,mode="DIST"):
        if mode == "MAX": #Mode for selecting max-Q action
            Q_s = self.Q[state][0] #Grab first Q in slice
            aM = Q_s[0]-1 #highest Q action holder
            act = [] #List of viable actions
            for a in range(self.A): #Looping over actions
                if aM < Q_s[a]: #If found a higher Q
                    act = [a] #Set action to candidate list
                    aM = Q_s[a] #Get new max Q
                elif aM == Q_s[a]: #If equal Q
                    act = act + [a] #Add actions to candidate list
            if len(act) == 0:#If no action found
                print(Q_s) #Print Q slice to find out why
            a_o = act[random.randint(0,len(act)-1)] #pick a random candidate
            self.history = [(state,a_o)] + self.history[:self.depth-1] #Add to history
            #self.history = self.history[1:] + [(state,a_o)] #Optional limited-depth history
            return a_o #Return action

        if mode == "DIST": #Mode for probability distribution based selection
            Q_s = self.Q[state][0] #Grab Q slice
            Q_M = self.Q[state][1] #Grab sum of CDF
            aCDF = Q_s[0] #grab the initial Q to build the CDF
            ri = random.random()*Q_M #Random number within CDF range
            i_a = 0 #action index
            cdf = [aCDF] #list format of CDF
            while ri > aCDF: #While not finding the prob. crossover for random value
                i_a+=1 #Increase index
                if i_a >= len(Q_s): #if less than length of slice
                    print(ri,cdf) #print values to see why
                    print(Q_s,Q_M,sum(Q_s))
                aCDF+=Q_s[i_a] #increase CDF
                cdf = cdf+[aCDF] #Add to list
            self.history = [(state,i_a)] + self.history[:self.depth-1] #Add action to history
            return i_a #Return action

    #Method to train 
    def train(self,R,_depth=None,_mode="STD"):
        if _depth != None: #If depth specified
            for pair in self.history[:_depth]: #For each S/A pair in history to that depth
                self.train_single(pair[0],pair[1],R,mode=_mode) #train on that pair with given R
        else: #Otherwise
            for pair in self.history: #Train on whole history
                self.train_single(pair[0],pair[1],R,mode=_mode)

    #Method to train on one instance S/A pair
    def train_single(self,state,act,R,mode="STD"):
        if mode == "STD": #Standard training method
            Q_s = self.Q[state][0]+[] #Grab previous Q value
            Qsa_n = Q_s[act] + self.alpha*(R - Q_s[act]) #Calculate update for S/A pair
            Q_s[act] = Qsa_n #Update Q array
            self.Q[state] = (Q_s,sum(Q_s)) #Update Q and summation list

        if mode == "STD_bias": #Standard training with bias
            Q_s = self.Q[state][0]+[] #Previous Q value
            Qsa_n = Q_s[act] + self.alpha*(R - Q_s[act]) #calculate update for S/A pair
            for a in range(self.A): #For each action
                Qsa_n = Q_s[a] + self.alpha*((1.0*self.Rmax-R) - Q_s[a]) #inverse train other actions
                Q_s[a] = Qsa_n #update Q
            self.Q[state] = (Q_s,sum(Q_s)) #update Q/sum list

        if mode == "SCALED": #Scaled training
            Q_s = self.Q[state][0]+[] #Previous Q
            for a in range(self.A): #For eahc action
                Qsa_n = Q_s[a]*(1+self.alpha)**(R*(2*(a==act)-1)) #update with scale factor
                Q_s[a]= Qsa_n #update Q
            self.Q[state] = (Q_s,sum(Q_s)) #update Q/sum list


#Standard sigmoid for IO normalization
def sigmoid_01(x,lmbda):
    return (1.0 / (1 + math.e**(-1.0*lmbda*x)))

#Class implementing standard Q Learning
class QLearning:

    def __init__(self,S,A,l,i=0):
        #Initialize state/action space

        self.sp = 0 #Previous states and actions
        self.ap = 0

        self.l = l #Learning rate
        self.S = S #State and action lists
        self.A = A

        #Initialization mode- 1 is random, otherwise weighted proportional
        if i == 0:
            self.Q = np.random.random((S,A))
        else:
            self.Q = (1.0/A)*np.ones((S,A))

    #Method to implement selecting an action
    def act(self,s):
        As = np.cumsum(self.Q[s,:]) #Cumsum over the state slice of Q array
        r = As[-1]*random.random() #random selection of number in cumsum
        sel = np.sum(1.0*(As < r)) #selection by sum of elements less than selected cumsum level
        self.sp = s #previous state set to state from which action taken
        self.ap = sel #previous action set to selected action
        return sel #Return the seleted action

    #Method to implement a training step
    def train(self,r,learn_law=None):
        #learn_law lets you put in a function of (Qp,l,r) that's not the basic learning rule
        
        Qp = self.Q[self.sp,self.ap] #Grab the current Q value

        if (learn_law != None): #For default learning scheme
            Qn = Qp + self.l*(r - Qp) #Update Q with standard rule
            self.Q[self.sp,self.ap] = round(Qn,3) #put into value- rounded for numeric stability

        else:
            Qn = learn_law(Qp,self.l,r) #apply alternate RL rule
            self.Q[self.sp,self.ap] = round(Qn,3) #put into value- rounded for numeric stability


#Murin-based core Q-Learning class (adapted for outside support of variables)
class QLearningM:
    
    def __init__(self,S,A,l,i=0):
        #Initialize state number, action number, learning rate, and init mode

        self.l = l #Learning rate

        self.S = S #State and action numbers
        self.A = A

        #Initialize Q array- if 0, random, otherwise action-weighted
        if i == 0:
            self.Q = np.random.random((S,A))
        else:
            self.Q = (1.0/A)*np.ones((S,A))

    #Method to take an action
    def act(self,s,mode=1):
        if mode: #probabilistic
            As = np.cumsum(self.Q[int(s),:]) #Cumulative sum of Q array on action slice
            r = As[-1]*random.random() #Random index
            sel = np.sum(1.0*(As < r)) #Get action
        else: #maximum likelihoof
            m = np.max(self.Q[s,:]) #get max
            sel = max([i*(self.Q[s,i]==m) for i in range(np.shape(self.Q[s:s+1,:])[1])]) #Select max likelihood action
        return int(sel) #return selection


    #Method to train core QL network
    def train(self,r,sp,ap,d):
        Qp = self.Q[int(sp),int(ap)] #Grab prior Q value
        Qn = Qp*(1-self.l*d) + self.l*d*(r) #Update Q value- Murin assumes standard learning
        self.Q[int(sp),int(ap)] = Qn*(Qn >= 0) #update Q matrix, threshold negative values

#Class implementing the full Murin algorithm on top of the minimal QL class
# this version implements State/Action augmentation, observed to be most efficient
class Murin:

    #Initialize learner
    def __init__(self,S,A,l,m):

        #State and action numbers
        self.S = S
        self.A = A
        
        #Array of state/action pairs- **not including augmentation**
        #   M = [[s1,s2...]
        #        [a1,a2...]]

        self.M = np.zeros((2,m))
        self.m = m #Memory length

        #Build modified Q array in subclass
        self.Q = QLearningM(S*A,A,l,1) #SxA array holds concatenated classes

    #Method to take an action
    def act(self,s,mode=1):
        ap = self.M[1,0] #Grab previous action from memory
        sA = s + ap*self.S #Calculate augmeted state as stride-indexed number
        aS = self.Q.act(sA,mode=mode) #Pull action from subclass
        self.M[:,1:] = self.M[:,0:self.m-1] #Update memory with new values
        self.M[0,0] = sA #Update memory array
        self.M[1,0] = aS
        return aS #Return the action

    #Method for training
    def train(self,r):
        #Loops over the memory depth, updating each prior learning pair
        # with a decay discounted rate based on how far back it occurred
        for a in range(self.m):
            self.Q.train(r,self.M[0,a],self.M[1,a],1.0/(a+1))

#Murin implementation of stat/state linking (usually less efficient than action linking)
class QMS:

    def __init__(self,I,S,A,m,l):

        #State space variables
        self.I = I #state/state index length (non-augmented states)
        self.S = S
        self.A = A

        self.m = m #Memory depth
        self.l = l #Learning rate

        #Augmented state arrays
        self.aS = I*S #state/state
        self.aA = S*A #state/action (for updates)

        #Q array and memory list
        self.Q = np.ones((self.aS,self.aA))
        self.M = np.ones((2,m))*-1

        self.s = 0 #Initial 0-state for init- highlights the bootstrap problem for S/S linking

    #Method to take an action
    def act(self,i):
        si = i + self.I*self.s #augmented state from previous state as stride-indexed value
        As = np.cumsum(self.Q[si,:]) #get cumulative sum over state slice
        r = As[-1]*random.random() #random selection
        sel = np.sum(1.0*(As < r)) #select out the action index

        self.M[:,1:] = self.M[:,:-1] #update the memory array
        self.M[0,0] = si
        self.M[1,0] = sel

        self.s = state = i #update the state
        return sel #return the selected action

    #Method to do a training step
    def train(self,r):
        d = 1.0 #Initial decay parameter
        for a in range(self.m): #across the histore
            sp = self.M[0,n] #grab state and action
            ap = self.M[1,n]
            Qp = self.Q[sp,ap] #original Q value
            Qn = Qp*(1-self.l*d) + self.l*d*(r) #Update Q value
            self.Q[sp,ap] = Qn*(Qn >= 0) #Threshold and update
            d = d*0.5 #exponentially decreasing decay

class inhibitor:
    #Inhibitor inherits from variable class, with the addition of an
    # access stack for controlling value updates, and an abstracted
    # output function for publishing those updates

    def __init__(self,_var,_fn=(),_args=()):
        self.var = _var # Variable storing value (inherits thread lock)
        self.function = _fn # Function for updating
        self.args = _args # Optional extra args for update fn
        self.current_user = None # User flag
        self.current_tier = None # Priority tier flag
        self.tier_stack = [] # Stack of awaiting users

    def set_value(self,val,user):
        # Function to attempt a value update
        while self.current_user != None and not self.current_user.this_thread.isAlive():
            # Broad net to catch release of variable from dead threads if not released
            # gracefully- works down stack until active user found or stack empty
            self.release(self.current_user)
            if self.current_user == None:
                break
        
        if self.current_tier == user.tier:
            # If same tier user, push update
            self.var.set_value(val)

            # Optional to spin off update function in new thread- slower than using
            #  OP buffer process
            #thread = threading.Thread(target=self.function, args=(self.args,val,))
            #thread.start()

            self.current_user = user # Set current user to be new current-tier caller
            return 1

        if self.current_tier != user.tier:
            # If not on the same tier, we need to update the stack
            
            if self.current_tier == None:
                # If there's no users for this inhibitor, populate stack and run update
                self.current_tier = user.tier
                self.current_user = user
                self.var.set_value(val)

                # Optional spinoff update thread
                #thread = threading.Thread(target=self.function, args=(self.args,val,))
                #thread.start()

                return 1

            if user.tier < self.current_tier:
                # In the case the new user's on a lower tier, push onto stack and
                # rn the update
                self.tier_stack = [(self.current_tier,self.current_user)] + self.tier_stack
                self.current_tier = user.tier
                self.current_user = user
                self.var.set_value(val)

                # Optional spinoff update thread
                #thread = threading.Thread(target=self.function, args=(self.args,val,))
                #thread.start()

                return 1

            if user.tier > self.current_tier:
                # If new user is at higher tier, ignore request entirely
                return -1

    def release(self,user):
        # Function to relinquish control of inhibitor, automatically called when self
        # is accessed and a dead thread is detected in the stack, but should be cleared
        # gracefully by owner to optimized performance
        if user == self.current_user:
            # If the releasing angent is the current owner
            if self.tier_stack != []:
                # If stack is not empty, Ppop user off the stack, update the user and
                # tier flags
                self.current_tier = self.tier_stack[0][0]
                self.current_user = self.tier_stack[0][1]
                self.tier_stack = self.tier_stack[1:]
            else:
                # Case for empty stack after release
                self.current_tier = None
                self.current_user = None
            return 1
        else:
            # If the user is not the current owner, ignore release request
            return -1


class subsumption_module:
    #Singular functional block module for subsumption, contains execution code, parameters,
    # access functions, variables, and tiering information. Modules act as the users of the
    # inhibitor variables.
    #
    #Note that multiple modules can occupy the same tier. If their variable access is disjoint,
    # then the operation is continued to be guatanteed stable, if they have shared variable
    # access, then race conditions may develop.

    def __init__(self,_fn,_args,_out_vars,_tier):
        #Function is the code to be executed within the module. This poses a potential
        # security risk! Args can be any type- including variables if necessary.
        #
        # out_vars is a sorted list of the variables this module uses.
        # Note: out_vars must not be actual vals- function call uses proxies! This enforces
        #  the write by tier protection. May implement strict enforcement at some point.

        self.function = _fn # Function code- may return value of the active flag if convenient
        self.args = _args # Non-op vars for the function
        self.out_vars = _out_vars # OP vars which will be written to, fn can refer to them
        self.tier = _tier # Module's tier
        self.this_thread = None # Current thread in which code is running
        self.active = variable(False) # Activity flag

        # Whether the module should be restarted by master process on thread death
        self.mode = 0

        self.proxies = None # Set of proxy vars (for auto-setup via OP buffer)

    def set_proxies(self,_proxies):
        # Just adds in a proxy set
        self.proxies = _proxies

    def set_mode(self,_mode):
        # Changes mode- mode determines if the module should be restarted by the master
        # manager if the module dies
        self.mode = _mode
        return 1

    def make_active_arg(self):
        # Adds in the modules activity flag to the arg set (convenience function)
        self.args = self.args + [self.active]
        return 1

    def set_thread(self,_thread):
        # Asssign the module a thread
        self.this_thread = _thread
        return 1

    def run(self):
        # Function to run the code in self.function.
        self.active.set_value(True) # Mark thread as active

        # If no proxy vars supplied, or thread is terminated, quit execution
        # otherwise, run code indefinitely
        while self.proxies != None and self.active.value:
            # Run the function with the op proxies and arguments, with module as user
            # val_cont is optional
            val_cont = self.function(self.args,self.proxies,self)
            if val_cont != None:
                # If funciton makes use of val_cont, use it to set active flag
                self.active.set_value(val_cont)

        return 1


class output_buffer:
    #Independent process for pushing op variable values to exterior acssets
    # typically less bogged down than in-situ publication
    #
    # Publishes all tasks on-period, preventing lockup that slows async update

    def __init__(self,_out_vars,_out_fns,_args,_period):
        # _args is for std. arguments, _out_vars is for output variables (i.e
        #   maps to an op function in _out_fns)
        # Remember, building the op gate creates proxies tied to each natural variable
        #   so feed those proxies to the subservient modules.
        # proxy only updates the root var (fed to out_fns) if it's a priority valid
        #   access. Modules should release inhibitors!

        self.out_fns = _out_fns # Functions for writing each output
        self.args = _args # sets of additional args to write functions
        self.out_vars = _out_vars # actual OP gating variables
        self.proxies = [] # List for proxy vars to be made on startup
        self.active = variable(False) # thread activity flag
        self.this_thread = None # OP buffer thread access
        self.period = _period # Period setting update frequency

        for var in self.out_vars:
            # Create list of proxies for feeding to modules, one for each OP var
            self.proxies = self.proxies + [inhibitor(var)]
            var.set_inhibitor(self.proxies[-1])

    def start(self):
        # Spin off the periodic update function in its own thread
        self.active.set_value(True) # Set thread activity flag

        # Create a new thread for the buffer process, spinning the 'run' method
        buffer_thread = threading.Thread(target=self.run, args=())
        self.this_thread = buffer_thread
        buffer_thread.start()

        return 1

    def run(self):
        # Update procedure to run in buffer's thread

        # Timer for updating on-schedule
        timer = time.time()
        while self.active.value:
            # As long as thread is active, update op vars every period
            if time.time()-timer < self.period:
                # Delay during off-cycle time
                pass
            else:
                # On period length, update timer and push var changes from
                # current values in proxies
                timer = time.time()
                for f in range(len(self.out_fns)):
                    # For each output function, push update from OP vars
                    func = self.out_fns[f] # get the fn
                    op_var = self.out_vars[f] # get the associated OP var
                    arg = self.args[f] # fetch extra args
                    # Spin off the var update in its own thread (for efficiency, parallel
                    # updating is performed- buffer inits updates, does not carry them out)
                    thread = threading.Thread(target=func, args=(arg,op_var.value,))
                    thread.start()

        return 1


class subsumption_master:
    #Primary subsumption initializer and manager- provides basic functionality for
    # all subprocesses, which tend to run independently. Mainly starts and stops processes, and
    # ensures stable execution of all priority tasks.

    def __init__(self,_op_vars,_op_fns,_args,_period = 0.01):
        # Includes list of modules, op vars and fns for buffer/ in-situ updates, and
        # sets of extra args needed for functions
        self.modules = [None] # List of all modules- sorted by tier
        self.tiers = 0 # number of tiers, updated on population
        self.threads = [] # List of all threads started by main process (module threads)
        self.active = variable(False) # Master thread activity flag
        self.this_thread = None # Thread for master controller
        self.proxies = [] # Set of proxy vars, for use by buffer

        # Create OP buffer- handles proxy spawning for all OP vars
        self.op_buffer = output_buffer(_op_vars,_op_fns,_args,_period)

        # Optional proxy-making block for use with in-situ updates (typically)
        #  slower than using OP buffer, due to async lockup
        #self.out_vars = _op_vars
        #self.out_fns = _op_fns
        #self.out_args = _args
        #
        #for v in range(len(self.out_vars)):
        #    var = self.out_vars[v]
        #    fn = self.out_fns[v]
        #    ar = self.out_args[v]
        #    self.proxies = self.proxies + [inhibitor(var,fn,ar)]
        #    var.set_inhibitor(self.proxies[-1])


    def add_module(self,module):
        # Add a module to the master process- can be run at any time

        # Populate the module's proxy list with its needed inhibitors
        module.set_proxies([var.inhibitor for var in module.out_vars])

        # Update tiers list in modules
        if module.tier > self.tiers:
            self.tiers = module.tier
            self.modules = self.modules + [None]*(self.tiers-len(self.modules)+1)

        # Add current module to master list of modules
        if self.modules[module.tier] == None:
            # If first on tier, create list
            self.modules[module.tier] = [module]
        else:
            # Otherwise add to pre-existant list
            self.modules[module.tier] = self.modules[module.tier] + [module]

        return 1

    def start(self):
        # Master start for initiating all modules, OP buffer, and main thread
        # note that this is init only, not the steady-state process!

        self.active.set_value(True) # Set own active flag

        # Start OP buffer using own method (adds thread)
        self.op_buffer.start()

        # Run through all modules, starting each in turn (starting with lowest-tier)
        for level in self.modules:
            for module in level:
                # For each module, assign an independent thread spinning its own
                # run method
                thread = threading.Thread(target=module.run, args=())
                module.this_thread = thread
                # Add module's ID and thread to the list thereof
                self.threads = self.threads + [(thread,module)]
                thread.start() # Start the module in its thread

        # Activate the primary process thread for the master manager
        main_thread = threading.Thread(target=self.run, args=())
        self.this_thread = main_thread
        main_thread.start()

        return 1

    def stop_all(self):
        # Killall function for subsumption system

        # Turn all module thread flags off 
        for modules in self.modules:
            for module in modules:
                module.active.set_value(False)

        # Turn off the op buffer
        self.op_buffer.active.set_value(False)

        # Turn off the master process
        self.active.set_value(False)

        return 1
        
    def run(self):
        # Master manager operation process, managing module threads,
        # activations, and thread process updates

        # As long as active flag is set, run forever
        while (self.active.value):
            th_fin = [] # Updated thread list

            # Examine all module threads
            for t in range(len(self.threads)):

                # Grab the thread and associated module
                thread = self.threads[t][0]
                module = self.threads[t][1]

                # If the thread is alive, just bypass it
                if thread.isAlive():
                    th_fin = th_fin + [(thread,module)]
                else:
                    # If the thread died, check whether it's supposed to re-loop (module's
                    # mode value says whether to restart or not)
                    if module.mode == 0:
                        # mode 0 (default) means restart, so make a fresh thread,
                        # assign it, and start, adding it into the thread list
                        thread = threading.Thread(target=module.run, args=())
                        module.this_thread = thread
                        th_fin = th_fin + [(thread,module)]
                    else:
                        # If mode is not 0, then the thread is let lie dormant
                        pass

            # Set main threads list with the updated version
            self.threads = th_fin + []

        return 1


###
#Assets related to state machine implementation
###

def thread_action(action,active):
    # A simple function handle for calling actions in separate threads
    action.do(active)

def thread_machine(state_machine):
    #A function wrapper for running the primary state machine in a separate
    # thread
    while not(state_machine.stopped.value):
        state_machine.run_step()        



class action:
    #Action class for running arbitrary function code in threaded loop
    # for performing tasks. Action code is parallel to root machine,
    # but updates after loop completion, so as to allow thread orphaning.
    # Therefor, non-atomic loops should be implemented in the state machine,
    # not in actions.

    def __init__(self,_function,_variables):
        #_function takes in the variables listed in _variables, and
        # outputs the values those variables should be after the action
        # is completed, writes to actual variables happen after function
        # is complete.

        self.function = _function
        self.variables = _variables

    def do(self,active):
        #Method to perform the actual code specifed in _function. Active
        # is a flag variable which allows higher processes to orphan the
        # action if the state changes during execution.

        # Code takes in non-mutable copies of the values in the supplied
        # variables, not the variables themselves
        vars_out = self.function([a.value for a in self.variables])
        if active.value == True:
            # Upon completion of _function, if the thread has not been
            # orphaned, then the variables supplied to the action are
            # given updated values from the function
            for a in range(len(self.variables)):
                self.variables[a].set_value(vars_out[a])
            # It is critical to note that the functions take in:
            #   [var1, var2, var3....]
            # and output updated copies in the same order- NOT the true variable
            # reference [this prevents unstable access]
        else:
            # If the thread is orphaned (active is set False), then
            # the thread dies quietly without altering the state variables
            pass
        return 1


class state:
    #Class containing a state which has a numeric label, a set of actions as
    # defined above, and a list of transition conditions and resultant states

    def __init__(self,_label,_transition_sel_function):
        #The transition selection function is a means to select from a list
        # of True transition conditions, should more than one be valid; intended
        # for future use with priority queuing, but fully functional otherwise

        self.label = _label
        self.subsequents = []
        self.conditions = []
        self.actions = []
        self.transition_sel_function = _transition_sel_function

    def add_action(self,action):
        # Add an action to the list of parallel code executors for this state
        self.actions = self.actions + [action]

    def add_transition(self,expression,result):
        # Add a transition condition phrased in terms of a logical expression on
        # the varriables in the state machine
        self.subsequents = self.subsequents + [result]
        # Conditions are listed in terms of result state, for efficiency
        if len(self.conditions) < result+1:
            self.conditions = self.conditions + [None]*(result+1-len(self.conditions))
        # The expression is stored in a list for internal evaluation in the machine
        C = expression
        self.conditions[result] = (len(self.subsequents)-1,C)
        return 1

    def remove_transition(self,result):
        #Excisement of a transition based on the resultant state

        if self.conditions[resule] != None:
            location = self.conditions[result][0]
            self.conditions[result] = None
            subs = self.subsequents
            subs = subs[:location] + subs[location+1:]
            self.subsequents = subs
            return 1
        else:
            return -1

    def check_conditions(self):
        #Function to evaluate the transition conditions for this state and return
        # a list of those resultant states foe which the transition is True

        met_set = []
        for a in range(len(self.subsequents)):
            result = self.subsequents[a]
            value = eval(self.conditions[result][1])
            if value:
                met_set = met_set + [result]
        return met_set

 
class state_machine:
    #Primary state machine object which acts as the main instantiation of the
    # whole system and manages actions, transitions, and execution

    def __init__(self,_stopped):
        #The state machine object contains the list of states, but not actions
        # or transitions themselves, which are possessed within the state objects

        self.S = 0
        self.states = []
        self.current_state = None
        self.stopped = _stopped
        self.this_thread = None

    def add_state(self,state):
        #Constructor function to add a state to the list within this object

        self.S = self.S + 1
        if state.label > len(self.states)-1:
            self.states = self.states + [None]*(state.label - len(self.states) + 1)
        self.states[state.label] = state
        return 1

    def set_state(self,state):
        # Function to change the current state manually
        self.current_state = state
            
    def run_step(self):
        #Central operation method for the state machine- a single full step of
        # state operation, in which the actions for the state are instantiated in
        # independent threads, while a loop monitors state variables and evaluates
        # transition conditions during execution. Once all such actions are
        # completed, the primary loop ends and allows for a new call of .run() to
        # act for the next state (whether or not it has changed)

        # Lists holding he thread handles for actions, and their active flags
        thread_acts = []
        thread_actives = []

        # Populate thread lists with actual thread processes and flag variables
        for act in self.current_state.actions:
            thread_actives = thread_actives + [variable(True)]
            thread_acts = thread_acts + [threading.Thread(target=thread_action, args=(act,thread_actives[-1],))]

        # Initiate all actions
        [thread.start() for thread in thread_acts]

        # Primary operational block
        truncated = False # Flag for mid-loop cancelation
        check_once = False # Flag to ensure at least one condition check per state-loop
        next_state = self.current_state # Marker for next state- defaults to no transition
        while (check_once == False) or (sum([a.isAlive() for a in thread_acts])>0) and not(self.stopped.value):
            #Loop runs checking conditions for the state until all action threads die, or
            # the loop is canceled by a state transition. check_once is used to ensure
            # that even if all threads execute completely before the loop begins.

            check_once = True # Immediately clear single-run flag
            transitions = self.current_state.check_conditions() # Evaluate the transition conditions
            if len(transitions) > 0:
                # If there are transitions, the transition_sel_function specified for the
                # current state picks from among them the next state
                next_state = self.current_state.transition_sel_function([self.states[a] for a in transitions])
                if next_state != None:
                    # On occasion of a state transitions, all actions in progress are
                    # orphaned
                    for t in thread_actives:
                        t.set_value(False)
                    truncated = True
                    break
                else:
                    pass
            else:
                pass

        # Upon completion of the run loop, a final check of conditions is made if
        # the loop was not ended by a noted transition, to ensure changes made by
        # late terminus actions are caught
        if not truncated:
            transitions = self.current_state.check_conditions()
            if len(transitions) > 0:
                next_state = self.current_state.transition_sel_function([self.states[a] for a in transitions])

        # If the loop was terminated by suppression of the state machine, all threads are
        # orphaned so as to ensure garbage collection
        if self.stopped.value:
            for t in thread_actives:
                t.set_value(False)
            next_state = None

        # Once all updates are final, the state is updated
        self.current_state = next_state
        return 1

    def start(self):
        #The start method instantiates a new thread with a running state machine in it, and
        # stores the thread handle so the process may be manipulated dynamically

        self.this_thread = threading.Thread(target=thread_machine, args=(self,))
        self.this_thread.start()
        return 1

    def stop(self):
        #Sets the process termination flag which kills the machien

        self.stopped.set_value(True)


if __name__ == '__main__' and False:
    #Index Case for State Machine: Coin-operated Vending Machine

    #An example state mahcine implementation which emulates a simple
    # vending machine's operation, taking in a 'coin', requesting a drink
    # selection, tracking drink stores and having an 'out of drinks' and
    # cancellation state path.

    #Demonstrates the following:
    #    - Functionality (naturally)
    #    - Systemic concurrency
    #    - State machine design components
    #    - Cross-platform interaction when used with demo panel interface
    #    - Construction of states, actions, and transitions


    #Creation of variables for state machine use

    #Input Variables
    coin_register = variable(0)
    cancel_button = variable(0)
    drink_selection = variable("")

    #Internal Variables
    number_tea = variable(5)
    number_soda = variable(5)
    number_water = variable(5)
    outgoing_message = variable("")

    #Output Variables
    drink_dispensed = variable(0)
    out_drinks = variable(0)

    #General Purpose timer
    gp_timer = handy_assets.timer(1)

    #Action Functions
    # Definitions of functions for insertion into states' actions

    def requesting_coin_act(variables):
        # Function to request coin deposition
        outgoing_message = variables[0]
        outgoing_message = "Deposit Coin"
        return [outgoing_message]

    def waiting_coin_act(variables):
        # hold state waiting for coin
        time.sleep(0.1)
        gp_timer.set_time() # Note the use of the timer for the display delay- set in the preceeding
                            # state to when the delay occurs
        return []

    def receiving_coin_act(variables):
        # coin receipt registry
        coin_register = variables[0]
        outgoing_message = variables[1]
        outgoing_message = "Coin Received"
        coin_register = 0
        return [coin_register,outgoing_message]

    def requesting_drink_selection_act(variables):
        # display drink request
        outgoing_message = variables[0]
        outgoing_message = "Select Drink"
        return [outgoing_message]

    def awaiting_drink_selection_act(variables):
        # wait for drink selection
        time.sleep(0.1)
        return []

    def checking_drinks_act(variables):
        # check if requested drinks in stock [check implemented in transitions]
        time.sleep(0.1)
        return []

    def out_of_drinks_act(variables):
        # state indicating selected drink is expended
        out_drinks = variables[0]
        drink_selection = variables[1]
        outgoing_message = variables[2]
        out_drinks = 1
        outgoing_message = "Out of " + drink_selection
        return [out_drinks,drink_selection,outgoing_message]

    def dispensing_drink_act(variables):
        # action 'dispensing' te selected drink, i.e. decrementing storage counter
        drink_dispensed = variables[0]
        number_tea = variables[1]
        number_soda = variables[2]
        number_water = variables[3]
        drink_selection = variables[4]
        outgoing_message = variables[5]

        if drink_selection == "Tea":
            number_tea = number_tea - 1
        if drink_selection == "Soda":
            number_soda = number_soda - 1
        if drink_selection == "Water":
            number_water = number_water - 1
        outgoing_message =  "Enjoy Your " + str(drink_selection)
        gp_timer.set_time()
        drink_dispensed = 1

        return [drink_dispensed,number_tea,number_soda,number_water,drink_selection,outgoing_message]

    def wait_act(variables):
        # empty display-delay function, not that it is used repeatedly
        time.sleep(0.1)
        return []

    def clear_selection_act(variables):
        # reset selection after main process is complete
        drink_selection = variables[0]
        drink_dispensed = variables[1]
        outgoing_message = variables[2]

        outgoing_message = "Clearing Selection"
        drink_selection = "N"
        drink_dispensed = 0

        return [drink_selection,drink_dispensed,outgoing_message]

    def canceling_act(variables):
        # function initiating cancellation of current process
        cancel_button = variables[0]
        outgoing_message = variables[1]

        outgoing_message = "Canceling..."
        cancel_button = 0

        gp_timer.set_time()

        return [cancel_button,outgoing_message]

    def return_coin_act(variables):
        # Action for 'coin return' after cancellation
        outgoing_message = variables[0]
        outgoing_message =  "Have your coin back"
        gp_timer.set_time()
        return [outgoing_message]


    ###
    #Below the individual states are built by construction, addition of transitions, and
    # actions from above
    ###

    print("Building States")

    requesting_coin = state(0,lambda x: x[0])
    requesting_coin.add_transition("True",1) # Note that transitions can be any logical condition
    requesting_coin.add_action(action(requesting_coin_act,[outgoing_message]))

    waiting_coin = state(1,lambda x: x[0])
    waiting_coin.add_transition("coin_register.value > 0",2) # Transitions may use any variable
    waiting_coin.add_transition("cancel_button.value",7)
    waiting_coin.add_action(action(waiting_coin_act,[]))

    receiving_coin = state(2,lambda x: x[0])
    # Here the gp_timer is being used to control a display delay
    receiving_coin.add_transition("coin_register.value == 0 and gp_timer.check_time() > 1.0",3)
    receiving_coin.add_action(action(receiving_coin_act,[coin_register,outgoing_message]))

    requesting_drink_selection = state(3,lambda x: x[0])
    requesting_drink_selection.add_transition("True",11)
    requesting_drink_selection.add_action(action(requesting_drink_selection_act,[outgoing_message]))

    awaiting_drink_selection = state(11,lambda x: x[0])
    awaiting_drink_selection.add_transition("drink_selection.value != \"N\" ",8)
    awaiting_drink_selection.add_transition("cancel_button.value",7)
    awaiting_drink_selection.add_action(action(awaiting_drink_selection_act,[]))

    checking_drinks = state(8,lambda x: x[0])
    # This condition set is illustrative of compound conditions, as each result state
    # has only ONE total contition, necessitating logical conjunction
    # Also, it illustrates use of transitions to check internal conditions,
    # which could also be acheived within the action
    cond_to_6 = "(drink_selection.value == \"Tea\" and number_tea.value > 0)" + " or "
    cond_to_6 = cond_to_6 + "(drink_selection.value == \"Soda\" and number_soda.value > 0)" + " or "
    cond_to_6 = cond_to_6 + "(drink_selection.value == \"Water\" and number_water.value > 0)"
    checking_drinks.add_transition(cond_to_6,6)

    cond_to_5 = "(drink_selection.value == \"Tea\" and number_tea.value == 0)" + " or "
    cond_to_5 = cond_to_5 + "(drink_selection.value == \"Soda\" and number_soda.value == 0)" + " or "
    cond_to_5 = cond_to_5 + "(drink_selection.value == \"Water\" and number_water.value == 0)"
    checking_drinks.add_transition(cond_to_5,5)

    checking_drinks.add_action(action(checking_drinks_act,[]))

    out_of_drinks = state(5,lambda x: x[0])
    out_of_drinks.add_transition("True",9)
    out_of_drinks.add_action(action(out_of_drinks_act,[out_drinks,drink_selection,outgoing_message]))

    dispensing_drink = state(6,lambda x: x[0])
    dispensing_drink.add_transition("drink_dispensed.value == 1",12)
    dispensing_drink.add_action(action(dispensing_drink_act,[drink_dispensed,number_tea,number_soda,number_water,drink_selection,outgoing_message]))

    # Note that all 3 'waits' use the same action function, but are independent states
    # because they delay for different transitions. Could also be done within one general
    # 'wait', using state variables to select next transition- but this is cleaner
    wait1 = state(12,lambda x: x[0])
    wait1.add_transition("gp_timer.check_time() > 2.0",9)
    wait1.add_action(action(wait_act,[]))

    wait2 = state(13,lambda x: x[0])
    wait2.add_transition("gp_timer.check_time() > 1.0",10)
    wait2.add_action(action(wait_act,[]))

    wait3 = state(14,lambda x: x[0])
    wait3.add_transition("gp_timer.check_time() > 2.0",9)
    wait3.add_action(action(wait_act,[]))

    clear_selection = state(9,lambda x: x[0])
    clear_selection.add_transition("drink_selection.value == \"N\"",0)
    clear_selection.add_action(action(clear_selection_act,[drink_selection,drink_dispensed,outgoing_message]))

    # The cancel state illustrates using an action as its own gate control- cancel_button is
    # reset by the canceling_act function, so it is used to gate this state
    canceling = state(7,lambda x: x[0])
    canceling.add_transition("cancel_button.value == 0",13)
    canceling.add_action(action(canceling_act,[cancel_button,outgoing_message]))

    return_coin = state(10,lambda x: x[0])
    return_coin.add_transition("True",14)
    return_coin.add_action(action(return_coin_act,[outgoing_message]))

    #Build State Machine
    print("Building Machine")

    # machine_stop is the flag variable allowing remote closure of the main thread
    machine_stop = variable(False)
    vending_machine = state_machine(machine_stop)

    # All states must be added, otherwise there will be a missing reference
    print("Adding States")
    vending_machine.add_state(requesting_coin)
    vending_machine.add_state(waiting_coin)
    vending_machine.add_state(receiving_coin)
    vending_machine.add_state(requesting_drink_selection)
    vending_machine.add_state(checking_drinks)
    vending_machine.add_state(out_of_drinks)
    vending_machine.add_state(dispensing_drink)
    vending_machine.add_state(clear_selection)
    vending_machine.add_state(canceling)
    vending_machine.add_state(return_coin)
    vending_machine.add_state(awaiting_drink_selection)
    vending_machine.add_state(wait1)
    vending_machine.add_state(wait2)
    vending_machine.add_state(wait3)

    print("Starting Machine")
    # Initial state must be specified- on initialization, current_state is None!
    vending_machine.set_state(requesting_coin)
    vending_machine.start()

    ###
    #Closed-loop exterior access test
    # This test interacts with the Java panel interface to demonstrate the functionality
    # of the state machine and datashare in conjunction
    ###

    # Initiate a datashare server for this run [runs in separate thread]
    data_serv = handy_assets.datashare_server()
    data_serv.start()

    # Create a client for convenient interaction with the datashare
    client = handy_assets.datashare_client()
    client.write(['machine_stop'],[0]) #Initialize server with global stop off

    # We create a list of variables to act as inputs to the state machine
    in_var = [coin_register,drink_selection,cancel_button,machine_stop]
    in_var_share = ['coin_state','drink_state','cancel_state','machine_stop']

    # and a list to be outputs of the state machine
    out_var = [out_drinks,number_tea,number_soda,number_water,outgoing_message]
    out_var_share = ['outta_drinks','num_tea','num_soda','num_water','outgoing_message']
    # Note that the string labels must be consistent over all access instances!

    # the central process here translates IO for the state machine to datashare bridge
    while (not vending_machine.stopped.value):
        # Write the outputs to the datashare
        client.write(out_var_share,[a.value for a in out_var])
        # Read the input variables from the datashare
        labels,values = client.read(in_var_share)
        for a in range(len(in_var)):
            # update the inputs to the actual variable wrappers
            in_var[a].set_value(values[a])

    # Terminate server after exiting state machine
    client.stop()


