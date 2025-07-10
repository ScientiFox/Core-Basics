###
#EDA project data structures
# A set of data structures, and some structure specific algorithms
#
# Contents:
#    - Alias Tree
#    - Sorted List
#    - Linked List
#    - Union Tree
#    - Alley (Array-linked list)
#    - Queue
#    - Stack
#    - Priority Queue
#    - Graph
#
###

#Standards
import math,time,random

#Specific imports from other core library
import fundamental_algorithms

class alias_tree:
    #A structure which allows O(1) access to a list of aliases for an index
    # by mapping all sets of linked indices reflexively- each index points to a dealias
    # list, and each de-aliest list points to all its aliases. Data for each set is stored
    # in the labels. Is a 'tree' because it maps joined indices to the least index

    def __init__(self,):
        #Initialize- lists of aliases, dealiest lists, and labels
        self.aliases = []
        self.dealiased = []
        self.labels = []

    def add(self,label):
        #Method to add a new label
        index = len(self.aliases) #new index
        self.aliases = self.aliases + [index] #Add index ro alias list
        self.dealiased = self.dealiased + [[]] #Add a new dealias list
        self.labels = self.labels + [label] #Add label to index data

    def join(self,index_a,index_1):
        #Method to join two indices as aliased to one another

        #Get minimum and max between the two to join
        min_index = min([self.aliases[index_a],self.aliases[index_1]]) 
        max_index = max([self.aliases[index_a],self.aliases[index_1]])

        #get the dealias of the larger index
        max_dealiased = self.dealiased[max_index]
        for address in max_dealiased: #For each dealias
            self.aliases[address] = min_index #Add the max's aliases to the min's 

        #set max index's alias to the minimum
        self.aliases[max_index] = min_index
        #add the max index and its aliases to the min's
        self.dealiased[min_index] = self.dealiased[min_index] + [max_index] + max_dealiased
        self.dealiased[max_index] = [] #remove max index's dealiases

class sorted_list:
    #A list, which is kept perpetually sorted on insertion of new elements

    def __init__(self,_get):
        # self.get(item) is a function returning the comparable
        # member of item
        self.items = []
        self.get = _get

    def add(self,item):
        #Method to add an item

        if len(self.items) > 1: #If more than one item
            #location is searched in list
            loc = fundamental_algorithms.block_search(item,self.items,self.get,False)
            #patch item in
            self.items = self.items[:loc+1] + [item] + self.items[loc+1:]
        elif len(self.items) == 1: #If only one item so far
            if self.get(item) >= self.get(self.items[0]): #if it's greater, add it after
                self.items = [self.items[0],item]
            else: #if it's less, add it before
                self.items = [item,self.items[0]]
        else: #If no items, start the list
            self.items = [item]

    def remove(self,item):
        #Method to remove an item
        loc = fundamental_algorithms.block_search(item,self.items,self.get,False) #Find the 'nearest' item
        if self.items[loc] == item: #if it's the same item
            self.items = self.items[:loc] + self.items[loc+1:] #remove it
            return 1 #return 1 if found and removed
        else: #otherwise
            return -1 #return -1 if item isn't found

    def Print(self):
        #Function to pretty print a list
        Str = ""
        for i in self.items:
            Str = Str + str(self.get(i)) + "," #Print item and comparable metric
        Str = Str[:-1] #remove trailing column
        return Str


class union_tree:
    #Object for a union tree- simple, elegant, astoundingly useful

    def __init__(self):
        #Only member is the root
        self.root = None

    def get_root(self):
        #Just a wrapper
        return self.root

    def set_root(self,_root):
        #Another wrapper to set the root
        self.root = _root


class link:
    #Abstract link object-
    # atomic member of list, queue, ore stack

    def __init__(self,L,V):
        #Make a link object
        self.prev = -1 #previous ordered object
        self.post = -1 #following ordered object
        self.value = V #Value of the item 
        self.label = L #Label for the item
        self.information = None #Any extra information

    def set_post(self,link):
        #Method to set the following link
        self.post = link

    def set_prev(self,link):
        #Method to set the prior link
        self.prev = link

    def Print(self):
        #Function to pretty print this link
        bL = self.prev
        if bL != -1:
            bL = bL.label
        aL = self.post
        if aL != -1:
            aL = aL.label
        s = "(" + str(self.label) + "," + str(self.value) + ")"
        return s

class linked_list:
    #Object to hold a linked list

    def __init__(self):
        #Initialize the list
        self.head = -1 #List head
        self.tail = -1 #List tail
        self.len = 0 #Length of the list

    def add(self,link):
        #Method to add a link to the list
        if self.tail == -1: #If no tail, empty list
            self.head = link #set the current link to both the head and tail
            self.tail = link
        else: #Otherwise
            link.set_prev(self.tail) #Append to the tail
            self.tail.set_post(link) #link tail to this one
            self.tail = link #update new tail

        #Increase length and return
        self.len = self.len + 1
        return 1

    def insert(self,after,link):
        #Method to insert a link after another one

        if after != -1 and after.post != -1: #If there's links on either side
            before = after.post #save succeeding chain

            before.set_prev(link) #link suceeding chain to current
            link.set_post(before) #link current to suceeding chain

            after.set_post(link) #link preceeding chain to current
            link.set_prev(after) #link current to preceeding chain

        #If at the tail, just add to list normally
        if after != -1 and after.post == -1:
            self.add(link)

        #if at the head
        if after == -1:
            self.head.set_prev(link) #Add the current ahead of the head
            link.set_post(self.head) #link current to former head
            self.head = link #update the list head

        #Return when finished
        return 1

    def remove(self,link):
        #Method to remove a link
        if link != -1: #if an actual link (in case we snag before head or after tail)

            #These work connecting past the head because a -1 in either sets new as the head/tail
            if link.post != -1: #if there's a successor
                link.post.set_prev(link.prev) #connect successor to predecessor
            if link.prev != -1: #If a predecessor
                link.prev.set_post(link.post) #Add predecessor to successor

            #If after the update, either is now the head or tail, update those 
            if link.post == -1:
                self.tail = link.prev
            if link.prev == -1:
                self.head = link.post

            #Decrement length
            self.len = self.len - 1

            #Return when done
            return 1

        else: #Return -1 if nothing real to remove
            return -1

    def Print(self):
        #Method to pretty print the list, inheriting from link print
        s = ""
        c = self.head
        if c != -1:
            while c != self.tail:
                s = s + c.Print() + "*"
                c = c.post
            s = s + c.Print()
        return s

class Alley:
    #An Alley is an array/linked list joint structure
    # it allows indexing a linked list by array coordinates and
    # referring to a link's corresponding array location by a reflexive
    # indexing system

    def __init__(self,N):
        #Create array holder, insertion array, parallel linked list, and length
        self.array = [-1]*N
        self.insert_array = []
        self.list = linked_list()
        self.len = N

    def add(self,link):
        #Method to add a link

        if link.label > self.len: #If there's a label outside the current array length
            self.array = self.array + [-1]*(1+link.label-N) #Add in enough new entries to account
        self.list.add(link) #Add the link to the linked list
        self.array[link.label] = link #Index the link in the array

        #Find the link value in the insertion array
        loc = fundamental_algorithms.block_search(link.value,self.insert_array,lambda x:x.value,False)

        #insert the link in the insert array
        self.insert_array = self.insert_array[:loc+1] + [link] + self.insert_array[loc+1:]
    
    def remove(self,label):
        #Method to remove a label from the alley

        if self.array[label] != -1: #If the label is real
            link = self.array[label]#Grab the link from the array

            #Find the link location in the insertion array
            loc = fundamental_algorithms.block_search(link.value,self.insert_array,lambda x:x.value,False)
            n = 1 #index count
            while (self.insert_array[loc] != link): #While not at the exact link
                loc1 = loc - n #move back by n
                loc2 = loc + n #move forward by n
                if self.insert_array[loc1] == link: #If fount the link
                    loc = loc1 #update location
                elif self.insert_array[loc2] == link: #if found here
                    loc = loc2 #other location
                n+=1 #Otherwise, keep branching

            #Pull element from insert array
            self.insert_array = self.insert_array[:loc] + self.insert_array[loc+1:]
            self.array[label] = -1 #Remove label from array
            self.list.remove(link) #remove link from list

            return 1 #return 1 if successful
        else:
            return -1 #return -1 if item not found

    def insert(self,link):
        #Method to insert a link
        c = self.list.head #Grab the list head
        self.array[link.label] = link #Put link into array

        #If the list has a real head
        if c != -1:
            #locate the link in the insert array
            loc = fundamental_algorithms.block_search(link.value,self.insert_array,lambda x:x.value,False)
            loc = loc*(loc!=-1) #patch for altered block sort
            self.list.insert(self.insert_array[loc],link) #insert insert array entry to list
            #update insert array
            self.insert_array = self.insert_array[:loc+1] + [link] + self.insert_array[loc+1:]
        else: 
            self.list.add(link) #Add link diectly if nothing else present

        #Return when done
        return 1

class queue:
    #Object implementing a queue

    def __init__(self):
        #Initialize head, tail, and length
        self.head = -1
        self.tail = -1
        self.len = 0

    def push(self,link):
        #Method to push a link
        if self.head == -1: #If no head, list is empty
            self.head = link #link is both head and tail
            self.tail = link
            self.len = 1 #length is now 1
        else: #OtherwiseL
            link.set_post(self.head) #put prior head after current
            self.head.set_prev(link) #link prior head to current
            self.head = link #make current the new head
            self.len = self.len + 1 #increase length
        return 1 #return one when done

    def pop(self):
        #Method to pop an element off the queue
        if self.len > 1: #If more than one item
            out = self.tail #grab the tail
            self.tail = self.tail.prev #make prior of tail the new tail
            self.tail.post = -1 #set new tail's sucessor to -1
            self.len = self.len - 1 #reduce length
            return out #return popped link
        elif self.len == 1: #If only one item
            out = self.tail #grab 'tail'
            self.head = -1 #set head and tail to nothing
            self.tail = -1
            self.len = 0 #length is now zero
            return out #return popped link
        else: #Otherwise- empty queue, return -1 (consistent as null link marker)
            return -1

    def Print(self):
        #Method to pretty print queue
        s = ""
        c = self.head
        if c != -1:
            while c != self.tail:
                s = s + c.Print() + "*"
                c = c.post
            s = s + c.Print()
        return s

class stack:
    #Object implementing a stack

    def __init__(self):
        #initialize with head, tail, and null length
        self.head = -1
        self.tail = -1
        self.len = 0

    def push(self,link):
        #Method to push a link

        if self.head == -1: #So, no head?
            self.head = link #current becomes both
            self.tail = link
            self.len = 1 #length is one
        else: #Otherwise
            link.set_post(self.head) #Set former head to current sucessor
            self.head.prev = link #Link former head to current link
            self.head = link #update head to new link
            self.len = self.len + 1 #increase length
        return 1

    def pop(self):
        #Method to pop off stack

        if self.len > 0: #If not empty
            out = self.head #grab head
            self.head = self.head.post #set former second to new head
            self.head.prev = -1 #remove new head's predecessor
            self.len = self.len - 1 #decrease length
            return out #return popped link
        else: #Otherwise,
            return -1 #return nothing

    def Print(self):
        #Method to pretty print stack
        s = ""
        c = self.head
        if c != -1:
            while c != self.tail:
                s = s + c.Print() + "*"
                c = c.post
            s = s + c.Print()
        return s

class priority_queue:
    #Object implementing a priority queue

    def __init__(self):
        #Similar init to other lists, but with priority counts array
        self.head = -1
        self.tail = -1
        self.len = 0
        self.pr_counts = []

    def push(self,link):
        #Method to push onto the queue

        V = link.value #grab the link's value

        if self.len > 0: #If more than one in the queue
            curr = self.head #Grab the head
            while curr != -1 and curr.value <= V: #if not at end and value lower
                curr = curr.post #move along the queue

            if curr == self.head: #If the link goes at the head
                link.post = self.head #attach old head after link
                self.head.prev = link #link old head to current
                self.head = link #update head

            elif curr == -1: #if link goes at the end
                link.prev = self.tail #Add former tail as current predecessor
                self.tail.post = link #link former tail to current
                self.tail = link #update tail

            else: #otherwise, insert between two linkes
                link.prev = curr.prev #predecessor to current
                link.post = curr #current to predecessor
                curr.prev.post = link #Same for successor
                curr.prev.post = link
                curr.prev = link

        else: #If empty list
            self.head = link #Set head and tail to link
            self.tail = link

        #Increase length
        self.len = self.len + 1

        return 1 #Return one when done

    def pop(self):
        #Method to pop of the queue
        if self.len() > 0: #If not empty
            curr = self.tail #grab current from tail
            
            self.tail = curr.prev #update tail
            self.tail.post = -1

            self.len = self.len - 1 #Reduce length

            return curr #return link popped off

        else: #if empty
            return -1 #return -1

    def Print(self):
        #Method to pretty print the queue
        s = ""
        c = self.head
        if c != -1:
            while c != self.tail:
                s = s + c.Print() + "*"
                c = c.post
            s = s + c.Print()
        return s

class node:
    #Node container for graphs

    def __init__(self,_L,_V):
        self.label = _L #Node label
        self.value = _V #Node value
        self.predecessors = [] #predecessors, for directed
        self.successors = [] #successors, for directed
        self.inOrder = 0
        self.outOrder = 0

class edge:
    #edge container for graphs

    def __init__(self,_F,_T,_V):
        self.prior = _F #preceeding node
        self.post = _T #suceeding node
        self.value = _V #edge value/weight
        self.data = None #additional data, if needed

class graph:
    #Object representing a graph

    def __init__(self):
        #Initialization

        self.N = 0 #Number of nodes
        self.E = 0 #Number of edges
        self.nodes = [] #Node list
        self.edges = [] #Edge list
        self.connectivity = [] #connectivity array
        self.conn = None #For import-from-array below

    def dijkstra(self,root,goal=None):
        #Method to find minimum path tree in a graph
        # Returns:
        # code,edge_list
        # code- 1: good -1: disconnected subgroup

        #Arrays to hold nodes permanently assigned and edges in MPT
        perm_nodes = [root]
        MPT_edges = [None]*self.N
        MPT_edges[root.label] = -1

        #construct the list of edges on boundary as a sorted list
        edge_boundary = sorted_list(lambda x: x.value)

        for e in root.successors: #Add in initial edge boundary from root node
            edge_boundary.add(self.connectivity[root.label][e])

        #while not all nodes in MPT, and goal not found, if set
        while (len(perm_nodes)<self.N and perm_nodes[-1]!= goal):

            e = 0 #edge index in boundary

            #Looping over boundary edges until 
            while (e < len(edge_boundary.items)) and (e != -1):

                edge = edge_boundary.items[e] #grab eth edge
                if MPT_edges[self.nodes[edge.post].label] == None: #if edge successor node not in MST yet
                    MPT_edges[self.nodes[edge.post].label] = edge #Add edge for that node
                    perm_nodes = perm_nodes + [self.nodes[edge.post]] #Add that node to permanent list
                    edge_boundary.items = edge_boundary.items[e+1:] #cut boundary edges to those after this edge
                    e = -1 #not the next added edge was found
                else: #loop up until finding the next valid edge
                    e = e + 1

            #return block- covers all cases
            if (e == len(edge_boundary.items)): #if all boundary edges examined: MPT not done
                return -1,MPT_edges #Return a failure and the edges constructed so far
            if (len(perm_nodes) == self.N): #If all nodes added
                return 1,MPT_edges #return success and full MPT
            if (perm_nodes[-1] == goal): #If goal found
                return 1,MPT_edges #return success and MPT up to goal

            #If not done:
            #For each edge leading out of newly added node
            for n in perm_nodes[-1].successors:
                if MPT_edges[n] == None: #If edge doesn't lead to one already in MST
                    edge_boundary.add(self.connectivity[perm_nodes[-1].label][n]) #Add to boundary for consideration


    def make_from_array(self,conn):
        ###
        #Builds the self graph from a connectivity array
        #  where conn[v1,v2] is the value associated with the
        #  edge v1-v2. Note: build process is O(|E|)!
        #  Designed to accept many formats natively, presumes
        #  that in an nxn format, None is the indicator for
        #  non-linked nodes (allows 0 weight)
        ###

        self.conn = conn.copy() #copy connectivity from input
        n = len(conn) #get array side length

        #Build nodes
        for vi in range(n):
            nodei = node(vi,None)
            self.add_node(nodei)

        # Get build mode
        mode = ''
        try:
            len(conn[0][0]) #test for sequential pairing
            mode = 'seq'
        except:
            mode = 'ind' #otherwise indexed

        # Build edges
        if mode == 'seq':
            for vi in range(n): #loop over pairs
                for pair in conn[vi]: #grab a pair
                    vf = pair[0] #end node index
                    val = pair[1] #edge weight
                    edgeif = edge(self.nodes[vi].label,self.nodes[vf].label,val) #make edge
                    self.add_edge(edgeif) #add edge to graph
        elif mode == 'ind': #if indexed
            for vi in range(n):
                for vf in range(n): #loop ove each axis
                    if conn[vi][vf] != None: #if not null 
                        edgeif = edge(self.nodes[vi].label,self.nodes[vf].label,conn[vi][vf]) #make edge
                        self.add_edge(edgeif) #add edge
                    else:
                        pass #pass on null edge
        return 1 #return one when done
        
    def add_node(self,node):
        #Method to add a node to the graph
        self.N = self.N + 1 #increase node count
        if len(self.nodes) < node.label+1: #iF list too short
            addls = node.label + 1 - len(self.nodes) #Number to add
            self.nodes = self.nodes + [None]*addls #add empties to list
            self.connectivity = self.connectivity + [[]]*addls #Add connectivities to list
        self.nodes[node.label] = node #Add in node at label location

    def add_edge(self,edge):
        #Method to add an edge to the graph

        self.E = self.E + 1 #increase edge count
        self.edges = self.edges + [edge] #Add edge to list
        source = edge.prior #grab source from edge
        sink = edge.post #grab edge sink

        #If the source and sink are both already node-indexed
        if source < len(self.nodes) and sink < len(self.nodes):
            outs = self.connectivity[source] #fetch connectivity of the source
            if len(outs) < sink+1: #if fewer outs than sink index
                outs = outs + [None]*(sink+1-len(outs)) #Update source connectivity to account
            outs[sink] = edge #add the edge to the connectivity list
            self.connectivity[source] = outs #update source connectivity

            #Update successors of source with sink, and vice versa for the sink
            self.nodes[source].successors = self.nodes[source].successors + [sink]
            self.nodes[sink].predecessors = self.nodes[sink].predecessors + [source]

            #Increase outorder of source and inorder of sink
            self.nodes[source].outOrder = self.nodes[source].outOrder + 1
            self.nodes[sink].inOrder = self.nodes[sink].inOrder + 1

        else: #Return -1 if either node isn't in the graph
            return -1

###
# Testing
###

if __name__ == '__main___':

    ###
    #LQS test assets
    ###

    st = [] #list of links
    for a in range(10):
        r = random.randint(1,4) #A random number
        st = st + [link(a,r)] #build a link
        print("("+str(a)+","+str(r)+")") #display values

    #Make a queue, stack, linked list, and priority queue
    Q = queue()
    S = stack()
    L = linked_list()
    Pq = priority_queue()

    for a in st:
        Q.push(a)
        S.push(a)
        L.push(a)
        Pq.push(a)

    Q.pop()
    S.pop()
    L.pop()
    Pq.pop()

    print(Q.Print())
    print(S.Print())
    print(L.Print())
    print(Pq.Print())

    ###
    #Graph test block
    ###

    g_N = random.randint(5,50)
    g_conn = [[random.randint(0,20) for b in range(g_N)] for c in range(g_N)]

    G = graph()
    G.make_from_array(g_conn)

    code,MPT = G.dijkstra(G.nodes[0],goal=G.nodes[3])
    code,MPT = G.dijkstra(G.nodes[0])
