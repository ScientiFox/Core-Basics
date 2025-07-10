###
#EDA project combinatorics & heuristics
#
#  Contents:
#    - Genetic Algorithms
#    - LPT algorithm
#    - KNN classifier
#    - Knapsack alg.
#    - Kruskal's alg.
###

#Standards
import math,time,random

#Imports from other basics
import fundamental_algorithms,data_structures,math


class specimen:
    #A specimin class for genetic algorithms

    def __init__(self,_params,_ranges,_resolutions):
        #Initialization
        self.params = _params #set parameters
        self.ranges = _ranges #set value ranges
        self.resolutions = _resolutions #set genetic resolutions

        #Initial genes- random
        self.genes = [random.randint(0,2**(self.resolutions)-1) for a in range(self.n_param)]

    def to_params(self):
        #convert to binary params genes
        return [ranges[i]*(int(self.genes[i],2)/(2.0**(self.resolutions[i]))) for i in range(self.params)]


class genetic_algorithm:
    #Class to implement genetic algorithm

    def __init__(self,_fitness,_Rr,_P0,_params,_resolution,_mut_rate):
        #Set fitness function
        self.fn = _fitness

        #Set mixing parameters
        self.R_rate = _Rr
        self.mut_rate = _mut_rate
        self.population = [] #Population of solutions

        #set params and resolutions
        self.params = _params
        self.res = _resolution

    def crossover(self,s1,s2,locus):
        #Method to perform crossovers

        #Get binaries of locus areas (eg 11111000 00000111)
        bl = 2**(self.res)-2**(locus)
        br = 2**(locus)-1

        #generate two specimen copies of the inputs
        sR1 = specimen(s1.params,s1.ranges,s1.resolutions)
        sR2 = specimen(s2.params,s2.ranges,s2.resolutions)

        #For each parameter
        for a in range(self.params):
            g1 = s1.genes[a] #Grab the genes for each specimen
            g2 = s2.genes[a]

            gR1 = (g1&bl)|(g2&br) #Logical crossover
            gR2 = (g2&bl)|(g1&br)

            sR1.genes[a] = gR1 #New genes
            sR2.genes[a] = gR2

        #Return new specimins
        return sR1,sR2

    def mutate(self,s1):
        #Method for a mutation

        gR = [] #New genes list
        altered = False #If altered flag
        for a in range(self.params): #looping over params

            xor_val = 0 #XOR-ing flips in gene
            for b in range(s1.resolutions[a]): #for each bit
                ri = random.random() #random chance
                
                xor_val = xor_val<<1 #Shift current value over 1
                if ri < self.mut_rate: #If mutating
                    altered = True #note there's an alteration
                    xor_val+=1 #add 1 to the XOR

            gR = gR + [sR[a]^xor_val] #Update gene by XORing

        #If a mutation occurred
        if altered:
            sR = specimine(s1.params,s1.ranges,s1.resolutions) #Make new specimen
            sR.genes = gR #set new genes
            return sR #return specimen
        else:
            return None #else, no output


class job:
    #Holder class for shop scheduling tasks

    def __init__(self,_label,_duration,_pred,_succ):
        self.label = _label #job label
        self.duration = _duration #duration
        self.predecessors = _pred #any preecessors
        self.successors = _succ #any successors

def LPT(job_list,agents):
    #Function to solve job-shop scheduling with the Longest Processing Time algorithm
    work_lists = [[]]*agents #list of agent assignments

    #Sorted list of jobs
    sorted_list = fundamental_algorithms.quicksort(job_list,lambda x,y: x.duration > y.duration)
    t_fins = data_structures.sorted_list(lambda x:x[0]) #completion times- sorted list keeps workers sorted by availability

    #looping over the agents, add in first jobs
    for i in range(agents):
        work_lists[i] = work_lists[i] + [sorted_list[0]] #Add longest job
        t_fins.add((sorted_list[0].duration,i)) #add the duration to the completion times list
        sorted_list = sorted_list[1:] #Pop first element off list

    #while there's jobs to assign
    while len(sorted_list) > 0: 
        n_job = sorted_list[0] #grab next longest job
        sorted_list = sorted_list[1:] #pop off list

        n_agent = t_fins[0][1] #Grab next agent
        t_fins.items = t_fins.items[1:] #grab items from completion times list

        t_fins.add((n_job.duration,n_agent)) #Add next job to list
        work_lists[n_agent] = work_lists[n_agent] + [n_job] #Add job to agent assignments

    #Return the work lists for the agents
    return work_lists

class knn_point:
    #An object to represent a point for the KNN algorithm

    def __init__(self,_point,_k):
        #create the point
        self.neighbours = [] #List of neighbors
        self.k = _k #number of neighbors to track
        self.point = _point #point location
        self.max_distance = None #Max distance in neighbors- for dynamic algorithm

    def insert(self,_pt,di):
        #Method to insert a point

        if len(self.neighbours) < self.k: #If open neighbor slots
            i = block_search((_pt,di),self.neighbours,lambda x: x[1],False) #search by distance
            self.neighbours = self.neighbours[:i]+[(_pt,di)]+self.neighbours[i:] #add in neighbor
            self.max_distance = self.neighbours[-1][1] #update max disrance

        elif (di <= self.max_distance): #Otherwise- if closer than current max
            i = block_search((_pt,di),self.neighbours,lambda x: x[1],False) #search by distance
            self.neighbours = self.neighbours[:i]+[(_pt,di)]+self.neighbours[i:] #add in
            self.neighbours = self.neighbours[:-1] #remove excess neighbot
            self.max_distance = self.neighbours[-1][1] #update max distance

        else: #return nothing when not doing anything
            return 0

    def check_distance(self,_pt):
        #Method to check distance to another point- handles different-length points
        di = 0 #net distance
        i = 0 #index
        while (i < min([len(self.point),len(_pt.point)])): #Looping ove rleast dimensionality
            di = di + (self.point[i]-_pt.point[i])**2 #add dimensional distance to sum
            i+=1 #increment index
        di = di**0.5 #take square root
        self.insert(_pt,di) #try insert with distance

class KNN:
    #Class to handle a knn classifier with afore points object

    def __init__(self,_k,_dist):
        #Initialize
        self.points = [] #list of points
        self.k = _k #neighborhood size

    def add_point(self,_pt):
        #function to add a point
        pt = knn_point(_pt,self.k) #build point object
        for pnt in self.points: #for each point already in
            pt.check_distance(pnt) #check distance for new and check point
            pnt.check_distance(pt) 
        self.points = self.points + [pt] #add to points list

    def insert_points(self,_PTS):
        #Insert a set of points (assumed already classified)
        for pnt in _PTS: #for each point
            pt = knn_point(pnt,self.k) #make a point
            self.points = self.points + [] #copy array

    def find_points(self,_pt):
        #Method to locate a sample point's neighbors, without adding it reflexively
        pt = knn_point(_pt,self.k) #Make a point
        for pnt in self.points: #loop over all points
            pt.check_distance(pnt) #check distance and add to new point's list
        return pt #return the point
        

class item:
    #Item object for knapsack problem

    def __init__(self,_weight,_value,_limit):
        #Basically a struct
        self.weight = _weight #object weight
        self.value = _value #object value
        self.limit = _limit #limit of how many can be added

def get_pack_w_v(pack,g_item):
    #Function to get the weight and value of a given pack
    # pack := [(item,ct),(item,ct)...]

    w = 0 #weight and value
    v = 0

    for a in range(g_item): #looping over pack contents
        item = pack[a][0] #grab ath item
        ct = pack[a][1] #grab number of items
        w = w + item.weight*ct #add weight to tally
        v = v + item.value*ct #add value to tally

    return w,v #return weight and value

def knapsack_heuristic(items,capacity):
    #Heuristic algorithm for knapsack algorithm 

    item_pr_val = [(it.value/it.weight,it) for it in items] #value to weight ratio for all items

    #quicksort items by value to weight ratio
    it_by_hr = fundamental_algorithms.quicksort(item_pr_val,lambda x,y:x[0]<y[0])

    #Make a fresh pack and set the capacity
    pack = []
    cap = capacity

    while len(it_by_hr) > 0: #while items remain
        item = it_by_hr[-1] #Grab first off stack
        num = min([math.floor(cap/item.weight),item.limit]) #Grab as many as you're allowed to fit in
        pack = pack + [(item,num)] #pack in that many
        cap = cap - num*item.weight #reduce the capacity by what we took up
        it_by_hr = it_by_hr[:-1] #pop the item off the stack

    #Return the finished pack
    return pack

def knapsack(items,capacity):
    #Actual dynamic knapsack algorithm
    # Presumes discrete capacity & weights

    #2d packs table
    table = [[None for a in range(capacity+1)] for b in range(len(items))]

    #Final weights, values, and plan variables
    fin_v = -1
    fin_w = -1
    fin_plan = []

    #For each weight in the table's rows
    for weight in range(len(table[0])):
        #find the max number of item_0s that can go in
        nmax = min([int(math.floor(weight/items[0].weight)),items[0].limit])
        val = nmax*items[0].value #Calculate the value thereof
        W = nmax*items[0].weight #calculate the weight
        table[0][weight] = (W,val,[(0,nmax)]) #Add the weight, value, and inital packing to table
        if val > fin_v: #If the value is higher than the current max, update final pack config
            fin_v = val
            fin_w = W
            fin_plan = [(0,nmax)]

    #Now, ofr each item in the columns after the first one
    for item in range(len(items))[1:]:
        #For each maximum weight in rows
        for weight in range(len(table[item])):
            #Get the max number of items that can fit into the residual capacity
            nmax = min([int(math.floor(weight/items[item].weight)),items[item].limit])
            v_max = -1 #holders for intermittent value and weight
            w_max = -1
            plan_max = [] #local (column-wise) max plan
            for a in range(nmax+1): #looping over max number of these items
                wc = weight - a*items[item].weight #calculate capacity loss from this item selection
                val = table[item-1][wc][1] + a*items[item].value #add in value for this many of them
                if val > v_max: #If it's an improvement, update the column max
                    v_max = val #new value
                    plan_max = table[item-1][wc][2] + [(item,a)] #new plan
                    w_max = table[item-1][wc][0] + a*items[item].weight #new weight
            table[item][weight] = (w_max,v_max,plan_max) #put the local max value on this table slot
            if v_max > fin_v: #if the local max is better than the global, update the global
                fin_v = v_max
                fin_w = w_max
                fin_plan = plan_max

    #At the end, we iteratively built the optimal global, so return that
    return fin_v,fin_w,fin_plan,table


###
#Kruskal's algorithm for MST of a graph
# (using streamlined Union-find)
###

def deep_root(u_tree):
    #Find the deep root from a given union tree
    updates = [] #list of updates
    next_up = u_tree #next check
    while next_up.root != None: #While some root available
        updates = updates + [next_up] #add node to list
        next_up = next_up.root #move to root
    return next_up,updates #return final root and list of chain to it

def kruskal_S(G):
    #Kruskal's algorithm with built-in edge sort
    # sorts edges, speeinf up the algorithm substantially
    E = fundamental_algorithms.quicksort(G.edges,lambda x,y: x.value < y.value)
    return kruskal_no_sort(G,E) #return kruskal on the sorted edge list

def kruskal_no_sort(G,E):
    #Uses kruskal's algorithm to generate the MST for G
    # E is the list of edges in G

    #Make a graph object
    MST = data_structures.graph()
    edges_sorted = E #edges

    #List of groups in G, at most N of them
    base_groups = [None]*G.N

    #While not all nodes in MST
    while MST.N < G.N:
        current_edge = edges_sorted[0] #Grab next edge

        #Get nodes edge connects
        node_A = G.nodes[current_edge.prior]
        node_B = G.nodes[current_edge.post]

        #Grab which subtrees each is in
        group_A = base_groups[node_A.label]
        group_B = base_groups[node_B.label]

        #If neither is grouped yet, make new group
        if (group_A == None) and (group_B == None):
            new_union = data_structures.union_tree() #Make a new union tree group
            base_groups[node_A.label] = new_union #Add to groups lookup
            base_groups[node_B.label] = new_union
            MST.add_node(node_A) #Add nodes to MST
            MST.add_node(node_B)
            MST.add_edge(current_edge) #Add edge to MST
        elif (group_A != None) and (group_B == None): #If A has a group, add B to it
            base_groups[node_B.label] = group_A
            MST.add_node(node_B)
            MST.add_edge(current_edge)
        elif (group_A == None) and (group_B != None): #If B has a group, add A to it
            base_groups[node_A.label] = group_B
            MST.add_node(node_A)
            MST.add_edge(current_edge)
        elif (group_A != None) and (group_B != None): If both have groups
            root_A,updates_A = deep_root(group_A) #Fetch deep roots of each
            root_B,updates_B = deep_root(group_B)
            if root_A == root_B: #If the roots are equal
                for a in updates_A+updates_B: #set the root of all progenitors to the deep root
                    a.set_root(root_A)
            elif root_A != root_B: #If not the same deep root
                root_new = data_structures.union_tree() #make a new tree
                root_A.set_root(root_new) #Add A and B to the new one
                root_B.set_root(root_new)
                for a in updates_A+updates_B: #Re-root all progenitors to new deep root
                    a.set_root(root_new)
                MST.add_edge(current_edge) #add the edge to the MST

        #Pop that edge off the queue
        edges_sorted = edges_sorted[1:]

    #Return the MST        
    return MST
        
###
#Algorithm tests
###


#Utility function to pretty print Graph edges
def print_edges_clean(G):
    g_N = G.N #Number of nodes
    for a in range(g_N): #For each node
        for b in range(len(G.connectivity[a])): #For all its neighbors
            if G.connectivity[a][b] != None: #If not disconnected
                print (min([a,b]),max([a,b])),G.connectivity[a][b].value #Print node order and edge length

if __name__ == '__main__':

    ###
    #Knapsack test
    ###

    #Make some items
    item1 = item(3,2,2)
    item2 = item(1,2,1)
    item3 = item(3,5,2)
    item4 = item(2,4,3)

    #Run the algorithm
    V,W,plan,table = knapsack([item1,item2,item3,item4],10)

    #Print value and weight, then plan
    print V,W
    print plan

    #Print the algorithm table
    for a in range(len(table)):
        for b in range(len(table[a])):
            print table[a][b][1]," ",
        print ""

    ###
    #Kruskal test
    ###

    #Holders for random variables
    data1 = [0]*51
    data2 = [0]*51
    cts = [0]*51

    #Looping 2000 times
    for a in range(2000):
        #Make a random-sized graph
        g_N = random.randint(5,50)
        g_conn = [[random.randint(0,20) for b in range(g_N)] for c in range(g_N)]

        #Make the graph
        G = data_structures.graph()
        G.make_from_array(g_conn)

        ti = time.time()#Initial time
        #Sort edge list
        E = fundamental_algorithms.quicksort(G.edges,lambda x,y: x.value < y.value) 
        tf1 = time.time() #Sort time
        G_mst = kruskal_no_sort(G,E) #Do Kriskal's algorithm
        tf2 = time.time() #Kruskal time

        #Grab output data
        data1[g_N] = data1[g_N] + (tf1-ti)
        data2[g_N] = data2[g_N] + (tf2-tf1)
        cts[g_N] = cts[g_N] + 1

        #Every 1000 trials
        if a%1000 == 0 and a != 0:
            print(a) #print a tick

    #Spacing
    print("")

    #Print output data
    for a in range(len(data1)):
        print(a**2,data1[a],data2[a],cts[a])


