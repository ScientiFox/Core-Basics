###
#EDA project fundamental algorithms
#
#Contents:
#    - Quicksort
#    - log_2 block search
#
###

#Standards
import math,time,random

def block_search(key,items,get,verbose):
    #Block-based search, log time
    #'get' is a function that returns the search relation value 

    #Build relation function
    relate = lambda a,b: -1*(get(a)<get(b)) + 1*(get(a)>get(b))

    L = len(items) #Length of list
    D = L/2 #Initial distance
    pos = int(L/2) #initial position

    if relate(key,items[0]) == -1:
        return -1 #Fail if key not in the list

    if relate(key,items[0]) == 0:
        return 0 #Return 0 if at the 0 index

    if relate(key,items[-1]) == 1 or relate(key,items[-1]) == 0:
        return len(items)-1 #return last index if item there or after

    while True: #Run until done

        #if at item location
        if relate(key,items[pos]) == 0: 
            return pos #return pos

        #If between less and greater items
        elif relate(key,items[pos]) == 1 and relate(key,items[pos+1]) == -1:
            return pos

        #If less than item
        elif relate(key,items[pos]) == -1:
            D = int(D/2) + 1*(int(D/2) == 0) #Half distance, minimum 1
            pos = pos - D #slide back by D

        #If greater than item
        elif relate(key,items[pos]) == 1:
            D = int(D/2) + 1*(int(D/2) == 0) #Half distance, minimum 1
            pos = pos + D #Slide forward by D

def quicksort(items,relate):
    #Function implementing quicksort
    # Note that 'relate' is a lambda function providing
    #  general access to the relation between two set members
    #  via relate(A,B) = 1 if A < B and 0 if A >= B.
    #
    # items is an unsorted array of objects

    if len(items) > 1: #If more than one item
        L = len(items) #Get length
        pivot = int(L/2) #Pivot around halfway point

        L1 = [] #Upper and lower lists
        L2 = []

        for a in range(pivot): #for items up to pivot
            if relate(items[a],items[pivot]): #if less than pivot
                L1 = L1 + [items[a]] #add to lower list
            else: #otherwise
                L2 = L2 + [items[a]] #add to upper list

        #For everything obove the pivot
        for a in range(len(items))[pivot+1:]:
            if relate(items[a],items[pivot]): #if less than pivot
                L1 = L1 + [items[a]] #Add to lower
            else: #oterwise
                L2 = L2 + [items[a]] #Add to upper

        #Recursively sort lower and upper lists
        L1s = quicksort(L1,relate)
        L2s = quicksort(L2,relate)

        #return sorted lower, plus pivot, plus sorted upper, together sorted
        return L1s + [items[pivot]] + L2s
        
    else: #If one item
        return items #return it! It's sorted!


###
#Validation block
###

if __name__ == '__main__':

    #Number of tests
    test = 20001
    correct = 0 #Number correct

    #Testing block values and counts
    T_block = [0]*1001
    T_block_ct = [0]*1001

    #Output data and length counts
    data = [0]*1001
    cts = [0]*1001

    #Looping over number of tests
    for i in range(test):
        #Number ranges
        Nl = 5 #low
        Nh = 1000 #high

        #Low and high values
        vl = random.randint(0,10)
        vh = random.randint(15,100)

        #Length and list generartions
        Ln = random.randint(Nl,Nh)
        itm = [random.randint(vl,vh) for a in range(Ln)]

        #Original list
        itm_o = itm + []

        #Do quicksort, timed
        t1 = time.time()
        srtd = quicksort(itm,lambda a,b: a<b)
        t2 = time.time()

        #Add time to data and increment length counts
        data[Ln] = data[Ln] + t2-t1
        cts[Ln] = cts[Ln] + 1

        #Random search key
        key = random.randint(min(srtd),max(srtd))

        #Do block search, timed
        ti = time.time()
        index = block_search(key,srtd,lambda x:x,False)
        tf = time.time()

        #Update times and Length counts for search
        T_block[Ln] = T_block[Ln] + (tf-ti)
        T_block_ct[Ln] = T_block_ct[Ln] + 1

        #Update search correct counter
        if (srtd[index] == key) or (srtd[index] < key and srtd[index+1]>key):
            correct+=1
        else: #If a failure, print condiitons for debug
            print(key,srtd)

        #Check against built-in sort
        #itm_o.sort()
        #if srtd == itm_o:
        #    correct = correct + 1

        #Print iterative results throughout
        if i%100 == 0 and i != 0:
            print(i,str(round((100.0*correct)/(i+1),2))+"% correct")

    #Print data at end of epoch
    for a in range(len(data)):
        print(a,data[a],cts[a])




