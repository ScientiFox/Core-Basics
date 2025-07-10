###
#EDA neural networks package
#
#Contents:
#
#   - Error Backpropagation
#   - Autoencoder
#
###

#Standards
import math,time,random

#For linear algebra
import numpy as np


class L3_EBP:
    #Standard EBP implementation
    #Assuming column vectors:
    #Wx = [OxI]*[Ix1]
    #
    #FF given by:
    #
    #i = x [or s(x)?]
    #h = s(Wx + b1)
    #o = s(Vh + b2)

    def __init__(self,_I,_H,_O,_a,_l,_eps=2.0,_mode=0):
        #Start up network

        self.a = _a #Learning update rate
        self.l = _l #feedforward scale rate
        self.I = _I #Input layer
        self.H = _H #Hidden layer
        self.O = _O #Output layer
        self.mode = _mode #Feedforward mode- 0 is direct, 1 is sigmoid normalized

        #Construct bias and weight array for input to hidden layer
        self.b1 = np.random.normal(loc=0.0,scale=_eps,size=(_H,1))
        self.W = np.random.normal(loc=0.0,scale=_eps,size=(_H,_I))

        #Construct bias and weights for hidden to output layer
        self.b2 = np.random.normal(loc=0.0,scale=_eps,size=(_O,1))
        self.V = np.random.normal(loc=0.0,scale=_eps,size=(_O,_H))

    def feedforward(self,x):
        #Feedforward function

        #Set input by mode- direct or normalized
        if self.mode == 0: 
            i = x
        elif self.mode == 1:
            i = 1.0/(1+math.e**(-1.0*self.l*x))

        #Calculate hidden layer transfer
        h = np.dot(self.W,i) + self.b1
        h = 1.0/(1+math.e**(-1.0*h)) 

        #Calculate output layer transfer
        o = np.dot(self.V,h) + self.b2
        o = 1.0/(1+math.e**(-1.0*o))

        #Return input, hidden activations, and output activations
        return i,h,o

    def step(self,x,y):
        #Single training step- makes updates, does not apply (for batch or single update)

        #FF step:
        i,h,o = self.feedforward(x) #Output
        e = y - o #Error

        #Diagnostic
        #print "h: ",h
        #print "o: ",o

        #Calculate hidden update layer vectors with EBP algorithm
        dy = -1.0*(y - o)*(o*(1-o))
        dh = (np.dot(self.V.T,dy))*(h*(1-h))

        #Diagnostic
        #print "dh: ",dh
        #print "dy: ",dy

        #Calculate matrix changes by back projection
        DV = np.dot(dy,h.T)
        DW = np.dot(dh,i.T)

        #Diagnostic
        #print "DW: ",DW
        #print "DV: ",DV

        #Bias weights are just the layer EBP vectors
        db2 = dy
        db1 = dh

        #Diagnostic
        #print "db1: ",db1
        #print "db2: ",db2

        #Rate modified changes to weight matrices
        dWn = -1.0*self.a*DW
        dVn = -1.0*self.a*DV

        #Rate modified changes to bias vectors
        db1n = -1.0*self.a*db1
        db2n = -1.0*self.a*db2

        #Function prototype
        # dWn,db1n,dVn,db2n,e = NN_ebp_step.step(X[:,0:1],Y[:,0:1])
        return dWn,db1n,dVn,db2n,e #Return array changes and measured error

    def train_batch(self,X,Y,N,verbose=0):
        #Method for batch training on XY array sets, through N cycles
        # Gets change for all samples, then update *after* all updates made
        # Faster and more efficient when appropriate, more likely to have excursions when not

        M = np.shape(X)[1] #Number of X training samples

        for n in range(N): #Looping over number of training cycles
            dWm = np.zeros(np.shape(self.W)) #initialize all array changes
            dVm = np.zeros(np.shape(self.V))
            db1m = np.zeros(np.shape(self.b1))
            db2m = np.zeros(np.shape(self.b2))

            eS = np.zeros(np.shape(Y[:,0:1])) #Squared error list

            for m in range(M): #For each X sample
                dWn,db1n,dVn,db2n,es = self.step(X[:,m:m+1],Y[:,m:m+1]) #Get one update
                dWm = dWm + dWn #Modify all holding arrays
                dVm = dVm + dVn
                db1m = db1m + db1n
                db2m = db2m + db2n
                eS = eS + es**2 #update squared error

            #Verbose status print
            if verbose>0 and n%verbose==0 and n!=0:
                print(n) #cycle number
                print((1.0/M)*eS) #Amortized error
                eS = np.zeros(np.shape(Y[:,0:1])) #reset cycle error

            #Actually apply batch weight updates
            self.W = self.W + (1.0/M)*dWm
            self.V = self.V + (1.0/M)*dVm
            self.b1 = self.b1 + (1.0/M)*db1m
            self.b2 = self.b2 + (1.0/M)*db2m

    def train_step(self,X,Y,N,order=0,verbose=0):
        #Method to do iterative training on a set of input vectors
        # optional in versus out of order training, can reduce bias using latter,
        # but costs possible temporal information

        M = np.shape(X)[1] #number of input samples

        eS = np.zeros(np.shape(Y[:,0:1])) #Squared errors

        for n in range(N): #For the number of training samples 
            if order == 0:  #For in-order training
                index = n%M #index loops over X in order
            elif order == 1: #If not in order, random index every time
                index = np.random.randint(0,M-1)

            #Grab individual training data slices
            x = X[:,index:index+1]
            y = Y[:,index:index+1]

            #Calculate single step updates
            dWn,db1n,dVn,db2n,es = self.step(x,y)

            #Update weights directly (after each training- influences output of later training)
            self.W = self.W + dWn
            self.V = self.V + dVn
            self.b1 = self.b1 + db1n
            self.b2 = self.b2 + db2n

            eS = eS + es**2 #Update squared error

            #FOr verbose mode, print cycle number and amortized squared error
            if verbose > 0 and n%verbose==0 and n != 0:
                print(n)
                print((1.0/verbose)*eS)
                eS = np.zeros(np.shape(Y[:,0:1])) #reset squared error list

class autoencoder:
    #Object implementing a simple autoencoder (astoundingly useful for data analysis, input compression, etc.)

    def __init__(self,_D,_H,_a,_l,_order):
        #Initialization

        self.D = _D #data layer
        self.H = _H #hidden layer
        self.a = _a #Learning rate
        self.l = _l #feedforward scale
        self.ord = _order #training mode (ordered versus random)
        self.NN = L3_EBP(_D,_H,_D,_a,_l,_mode=1) #Baseline EBP network backbone

    def train(self,X,N,verbose=0):
        #Specialized training wrapper- formats for X to X training

        M = np.shape(X)[1] #Input sample number
        Yx = 1.0/(1+math.e**(-1.0*self.l*X)) #'Output' is normalized input (for stability)
        self.NN.train_step(X,Yx,int(N*M),order=self.ord,verbose=verbose)  #Run a training step

    def ff(self,x):
        #Feedforward wrapper
        i,h,o = self.NN.feedforward(x) #Use backbone FF
        return i,h,o #Return input, hidden, and output layer- hidden is the 'real', useful output

###
#Test and validaiton block
###


if __name__ == '__main__':

    #get data- IRIS is a good test
    data_file = open("iris.data",'r')
    lines = data_file.readlines() #Read in data

    #Process string to numbers from CSV
    data_string = [l.split(",") for l in lines] 
    data_list = [[float(i) for i in l[:-1]] + [l[-1]] for l in data_string]

    del data_list[-1] #Remove trailing line

    classes = {} #Sorted classes
    Xs = [] #Sample data holders
    Ys = []

    #Looping over each sample, build IO vectors
    for sample in data_list:
        if (sample[-1] in classes): #Add a class if missing from set so far
            Ys = Ys + [classes[sample[-1]]]
        else: #Otherwise,
            v = np.zeros((3,1)) #Make vector for class
            v[len(classes)][0] = 1
            classes[sample[-1]] = v #Add to class list
            Ys = Ys + [classes[sample[-1]]] #Add to training output

        #Build the X input vector
        Xs = Xs + [np.array([sample[:-1]])]

    #EMpty main arrays for input to training
    X = np.zeros((len(data_list[0][:-1]),len(Xs)))
    Y = np.zeros((3,len(Ys)))

    #Looping over samples
    j = 0
    L = range(len(Xs))
    while len(L) > 0:
        r = random.randint(0,len(L)-1) #grab a random index
        i = L[r] #Grab selected index
        L = L[:r] + L[r+1:] #pop out of list
        X[:,j] = Xs[i] #Add sample to proper input array
        Y[:,j:j+1] = Ys[i] #Add training output to output array
        j+=1 #Increment counter

    N_test = 20 #Number of tests to run
    M_test = np.shape(X)[1] #get length of input data
    sub_n = int(0.8*M_test) #Subsample for out group testing
    subsample = (X[:,0:sub_n],Y[:,0:sub_n]) #Build the subsample 
    validate = (X[:,sub_n:],Y[:,sub_n:]) #build the validation set

    #Output data arrays
    sum_r_err = np.zeros(np.shape(X[:,0:1]))
    sum_err_ratio = np.zeros(np.shape(X[:,0:1]))
    sum_scale = np.zeros(np.shape(X[:,0:1]))

    N_exp = 30 #Number of experiments to run

    for q in range(N_exp): #Looping over each experiment

        #Announce new experiment
        print("New experiment")
        print("----------")

        #Make input, hidden, and output layer sizes
        I = np.shape(X[:,0:1])[0]
        H = 20 #Chosen for experiments
        O = np.shape(Y[:,0:1])[0]
        a = 0.9 #Learn rater
        l = 0.4 #normalization scale

        #Different testing options
        #NN_ebp_batch = L3_EBP(I,H,O,a)
        #NN_ebp_step = L3_EBP(I,H,O,a,l,_mode=0)
        NN_auto =  autoencoder(I,1,a,l,1)

        # Step train options
        #print("step_train: ")
        #e_unTrained = np.zeros(np.shape(Y[:,0:1]))

        #Error array
        e_unTrained_auto = np.zeros(np.shape(X[:,0:1]))

        for s in range(M_test-sub_n): #Looping over number to train on
            x = validate[0][:,s:s+1] #Check error on validation subset
            y = validate[1][:,s:s+1]

            #i,h,o = NN_ebp_step.feedforward(x) #standard feedforward
            ia,ha,oa = NN_auto.ff(x) #autoencoder feedforward

            # Update error for validation
            e_unTrained_auto = e_unTrained_auto + (ia-oa)**2 #autoencoder
            #e_unTrained = e_unTrained + (y-o)**2  #Standard EBP

        # Run a training step
        #NN_ebp_step.train_step(subsample[0],subsample[1],N_test*M_test,order=1,verbose=3*M_test)
        NN_auto.train(subsample[0],N_test,verbose = 0)

        # Update error for training
        #eS_step = np.zeros(np.shape(Y[:,0:1]))
        eS_step_auto = np.zeros(np.shape(X[:,0:1]))

        #Validate again after training
        op_avg = np.zeros(np.shape(X[:,0:1]))
        for s in range(M_test-sub_n):
            x = validate[0][:,s:s+1]
            y = validate[1][:,s:s+1]

            # FF for EBP or autoencoder
            #i,h,o = NN_ebp_step.feedforward(x)
            ia,ha,oa = NN_auto.ff(x)

            #Average amortized output
            op_avg = op_avg + oa*(1.0/(M_test-sub_n))

            #Error for autoencoder or EBP
            eS_step_auto = eS_step_auto + (ia-oa)**2
            #eS_step = eS_step + (y-o)**2

        # Cycle diagnostic display        
        print("Validate err:")
        print("_____________")

        # FOr EBP trials
##        print("EBP: ")
##        print("Ratio of post- to pre-train error:")
##        pT = (1.0/(M_test-sub_n))*e_unTrained
##        Pt = (1.0/(M_test-sub_n))*eS_step
##        print(Pt/pT)
##        
##        print("Raw error:"
##        print(Pt
##        print("_____________"

        #For autoencoder trials
        print("Autoencoder: ")
        print("Ratio of post- to pre-train error:")
        pT = (1.0/(M_test-sub_n))*e_unTrained_auto
        Pt = (1.0/(M_test-sub_n))*eS_step_auto
        print(Pt/pT)
        
        print("Raw error:")
        print(Pt)
        print("Avg scale: "+str(op_avg.T))
        print("_____________")

        sum_r_err = sum_r_err + Pt
        sum_err_ratio = sum_err_ratio + Pt/pT
        sum_scale = sum_scale + op_avg

        #Time delay or manual toggle options
        #time.sleep(3)
        #input()

    #Final test value outputs
    print("~~~~~~~~~")
    print("Avg. raw error:")
    for a in sum_r_err*(1.0/N_exp):
        print(round(a,4))

    print("Avg scale:")
    for a in sum_scale*(1.0/N_exp):
        print(round(a,4))
    
    print("Avg p-p ratio err (%):")
    for a in sum_err_ratio*(100.0/N_exp):
        print(str(round(a,2))+"%")

    print("ratio raw err to scale, avg (%):")
    for a in (sum_r_err/sum_scale)*100.0:
        print(str(round(a,2))+"%")


