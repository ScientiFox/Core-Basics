###
# EDA project handy assets
#  Useful miscellaneous objects and functions used for
#  general purpose operations
#
# Contents:
#  - Datashare Utility
#  - Timer
#  - Variable object
#
###

#Standards
import math,time,random

#For communications
import socket

#FOr parallelism
import threading

#Class to hold an arbitrary variable for sharing across containers
class variable:
    # Variable class wrapper for storing state data in thread-sharable
    # containers, along with access locking for read/write control

    def __init__(self,_init_value):
        # The lock is a member of the class, so it is shared with
        # threads inheriting variable access
        self.value = _init_value
        self.lock = threading.Lock()
        self.inhibitor = None

    def vcopy(self):
        #Function to make a copy of the object
        new_var = variable(self.value)
        new_var.inhibitor = self.inhibitor
        return new_var

    def set_inhibitor(self,inh):
        #Inhibit access from anywhere
        self.inhibitor = inh

    def get_value(self):
        # Non-locked because reading is non-interfering
        return self.value

    def set_value(self,v):
        # Lock is used to block concurrent access
        self.lock.acquire()
        try:
            self.value = v
        finally:
            self.lock.release()

#Packet object for transmission communications with the datashare
class packet:

    def __init__(self,_name,_mode,_length,_d_types,init_values,_interface,_units=None):
        #Packet init
        self.name = _name #A name for the packet 
        self.length = _length #Number of entries
        self.d_types = _d_types #array of data types
        self.units = _units #Units, if you need them
        self.types = [0.0,"",1] #Corresponding prototypes for float, string, and int
        self.interface = _interface #The interface over which the packet is sent 
        self.mode = _mode #0 peripheral 1 control

        #Holder arrays for float, string, and integer values. Fixed length for convenience
        self.f_vals = [0.0]*_length
        self.s_vals = ['']*_length
        self.i_vals = [0]*_length

        #Holder for the data
        self.data = (self.f_vals,self.s_vals,self.i_vals)
        for a in range(_length): #Load in initial values into the holder
            self.data[_d_types[a]][a] = init_values[a]

    #Method to set a value in the packet
    def set_value(self,index,val):
        if (type(val) == type(self.types[self.d_types[index]])): #select for type
            self.data[self.d_types[index]][index] = val #put in the right array
        else: #Don't insert incorrect data
            pass

    #Method to fetch data from a packet
    def get_val(self,index):
        return self.data[self.d_types[index]][index]

#Object for a packet transfer handler
class packet_exchange:

    def __init__(self,_rate=100,DSC=None):
        self.datashare = DSC #set the datashare it's attached to, if applicable
        self.casts = [float,str,int] #Casting types, in order

        #Threading variables
        self.this_thread = None #thread id
        self.stopped = False #flag to halt
        self.rate = _rate #Update rate
        self.clock = timer("pck_exg") #clock for the update rate

        #Holders for peripherals to send, the interfaces, and input index
        self.peripherals = {}
        self.p_interfaces = []
        self.receive_index = 0

        #Holders for the controls, there interfacers, and output index
        self.controls = {}
        self.c_interfaces = []
        self.transmit_index = 0


    #Add a packet for exchanging
    def add_packet(self,packet):
        if packet.mode == 0: #if peripheral packet
            self.peripherals[packet.name] = packet #Add
            if not(packet.interface in self.p_interfaces): #if no matching interface yet, add it
                self.p_interfaces = self.p_interfaces + [packet.interface]
        else: #If a control
            self.controls[packet.name] = packet #Add
            if not(packet.interface in self.c_interfaces): #If no matching interface, add it
                self.c_interfaces = self.c_interfaces + [packet.interface]

    #Receipt method - cycles through peripherals
    def receive(self):
        intf = self.p_interfaces[self.receive_index] #grab interface for the next input
        self.receive_index = (self.receive_index+1)%len(self.p_interfaces) #cycle index
        pack = intf.readline() #read from the interface
        pack = pack.split("&")[-1] #Split delimiter
        unpack = pack.split(",") #unpack the data

        #Grab the packet name
        p_name = unpack[0]

        #If the name is in the peripheral list
        if p_name in self.peripherals:
            packet = self.peripherals[p_name] #grab the packet
            if (len(unpack)-1 == packet.length): #check data in it (-1 for name)
                pass
            else:
                return -1 #otherwise, a fault
        else:
            return -1 #A fault if no peripheral for this packet, too

        i = 1 #parse index (i is out of the packet transmitted)
        while i < packet.length+1: #Loop over packet length (+1 after name)
            index = i-1 #Actual index in data arrays is  one less
            d_type = packet.d_types[index] #grab the type
            packet.data[d_type][index] = self.casts[d_type](unpack[i]) #cast to type and put in packet
            if self.datashare != None: #If there's a datashare
                #float,string,int -> int,float,string
                _type = (1+d_type)%3 #grab the right type
                self.datashare.soft_write(packet.name+str(index),_type,_data,packet.data[d_type][index]) #do a software write on the datashare
            i+=1 #Loop over packet

        return 1

    #Transmit method - cycles through controls
    def transmit(self):
        p_out = "&" #Delimiter character

        #grab the packet from the list of controls
        packet = self.controls[self.transmit_index]

        self.transmit_index = (self.transmit_index+1)%len(self.controls) #update transmit index

        #Loop over packet length
        index = 0
        while index < packet.length:
            p_out = p_out + str(packet.data[packet.d_types[index]][index]) #COnstruct packet string
            p_out = p_out + "," #separate with commas
            index+=1

        p_out = p_out[:-1] #Strip trailing comma
        p_out = p_out + "\n" #Terminate with newline

        packet.interface.write(p_out) #Write string to the interface

        #Return done
        return 1

    #Main run function
    def run(self):

        #As long as stopped flag isn't set
        while not(self.stopped):
            #Delay through update period
            while (self.clock.check_time() < (1.0/rate)):
                pass
            self.clock.set_time() #Update the clock
            self.receive() #check receipts
            self.transmit() #send controls
            
        return 1

    #Startup function
    def start(self):
        #Make a run thread
        self.this_thread = threading.Thread(target=self.run, args=())
        self.this_thread.daemon = True # Daemon thread executes past initiating process
        self.this_thread.start() #Start the thread

    #Stop function to call termination internally
    def stop(self):
        self.stopped = True
        return 1

class timer:
    #Implementation of a basic timer class to make checks
    # of duration since annotation simple

    #Start- clock has a label and an initial time
    def __init__(self,_label):
        self.label = _label
        self.timestamp = time.time()

    def check_time(self):
        # Return the time since last annotated stamp
        return time.time()-self.timestamp

    def set_time(self):
        # Mark the time for future elapsed time measurements
        self.timestamp = time.time()
        return 1


class datashare_client:
    #Data access client for the client/server datashare model. Rapid-access is only
    # guaranteed on local machine!

    def __init__(self,_port=50901):
        # Access is to local host, port can be canged but is defaulted to 50901, if
        # more servers are needed, then this must change
        self.ip = 'localhost'
        self.port = _port
        # Datastore records object type to give non-examination based access
        # less important for python, essential for C++ and Java
        # Note lack of boolean- due to weirdness in typecast, recommended to use ints
        self.types_ref = [type(1),type(1.1),type("one")]

    def read(self,labels):
        # Method to make a request for the data stored under the supplied labels
        request = "&," # requests start with '&', format specifed in the server
        for label in labels:
            request = request + label + ","
        request = request + "%" # all communications end with '%'

        # Open socket and send request
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect_ex((self.ip,self.port))
        s.send(request)
        data = s.recv(100)
        s.close()

        # Parse string response using type codes
        data_s = data.split(",")[1:-1]
        data_out = []
        for unpack in data_s:
            value = unpack.split(":")[0] # String of the data value
            type_ = unpack.split(":")[1] # String of the type code
            if type_ != "N": # Type key for 'no data' for a label requested
                value = self.types_ref[int(type_)](value)
            else:
                value = None # Data is 'none' if not stored
            data_out = data_out + [value] # Returns ordered list of the requests
        return labels,data_out
        
    def write(self,labels,data):
        request = "$," # Write requests begin with '$'
        # Formatting for writes in the client documentation
        # uses both a label and a data value
        for n in range(len(labels)):
            label = labels[n]
            value = data[n]
            # Get the type code
            type_ = sum([a*(type(value)==self.types_ref[a]) for a in range(len(self.types_ref))])
            request = request + label + ":" + str(type_) + ":" + str(value) + ","
        request = request + "%"

        # Open socket and send request
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.ip,self.port))
        s.send(request)
        data = s.recv(100)
        s.close()

        # no parsing for writes needed here
        return data

    def stop(self):
        # Send a simple packet which sets the internal stoppage flag for the
        # datashare
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.ip,self.port))
        s.send("*,%") # '*' is the stop code, message still ends with '%'
        data = s.recv(100)
        s.close()
        return data


class datashare_server:
    #Server object for datashare functionality, implements storage, encoding, and
    # lookup access functionality

    def __init__(self,_port=50901):
        # Access is to local host, port can be canged but is defaulted to 50901, if
        # more servers are needed, then this must change
        self.ip = 'localhost'
        self.port = _port
        self.buffer_size = 4096 # Server recv() buffer
        self.serv = None # Persistent server object from socket
        self.data = [] # Table of stored data values
        self.data_type = [] # Table of stored data types
        self.data_lookup = {} # Hash table of string labels
        self.types_ref = [type(1),type(1.1),type("one"),type(None)]
        self.stopped = False # Stoppage flag
        self.this_thread = None # Thread containing server process
        self.prev_data = None # Last data store- useful for displays

    def soft_write(self,label,_type,value):
        #$,label:type:value,label:type:value,...,%
        pck  = "$,"+label+":"+str(_type)+":"+str(value)+",%"
        self.parse_request(pck)

    def soft_read(self,label):
        #&,label,label,...,%
        pck = "&,"+label+",%"
        return self.parse_request(pck)

    def start(self):
        # Initiate a thread containing the datastore, running the main-loop code
        self.this_thread = threading.Thread(target=self.run, args=())
        self.this_thread.daemon = True # Daemon thread executes past initiating process
        self.this_thread.start()

    def stop(self):
        # Set stop flag, leading to thread termination
        self.stopped = True

    def run(self):
        # Main method running a loop which processes server requests and manages data

        # Create the socket server object
        self.serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv.bind((self.ip,self.port))
        self.serv.listen(4) # Set the server to listen, with a queue of 4 clients

        # Main loop executes until client sends .stop() message
        while not self.stopped:
            conn,addr = self.serv.accept() # Allow a connection
            packet = conn.recv(self.buffer_size) # Read in packet data
            reply = ''
            if packet != '': # If not empty data, parse the request
                reply = self.parse_request(packet)
            conn.send(reply) # Send whatever reply parse funciton generates
            
            #conn.close() # Currently, allows client side to end contact

            if self.data != self.prev_data:
                # Useful display of changed data only
                #print self.data
                self.prev_data = self.data + []

        print("closing")
        self.serv.close() # Closes once stopped fully

        return 0

    def parse_request(self,packet):
        #A method for actually parsing a request and executing the desired activity.
        # Pushes write commands to the datastore, and returns reads to the client.
        # Communication protocol formatting is given by:
        #
        #Packet:
        #        Writes-
        #        $,label:type:value,label:type:value,...,%
        #        return: $1%
        #
        #        Reads-
        #        &,label,label,...,%
        #        return: &,value:type,value:type,...,%
        #
        #        Stops-
        #        *,%
        #        return: *,%
        # $: write
        # &: read
        # *: stop

        spliced = packet.split(",") # Partition the packet by ','

        # Select parsing mode based on header character
        if spliced[0] == '$':
            write = True
        elif spliced[0] == '&':
            write = False
        elif spliced[0] == '*':
            self.stopped = True
            return "*,%"

        # Extract the actual request portion of the message
        request = spliced[1:-1]

        packet_resp = ""
        if write:
            # If writing, then for each given label
            for a in request:
                unpack = a.split(":")
                # Divide the label, type and value by ':'
                label = unpack[0]
                type_ = self.types_ref[int(unpack[1])]
                value = type_(unpack[2])
                # If the label exists, overwrite old value
                if label in self.data_lookup:
                    self.data[self.data_lookup[label]] = value
                    self.data_type[self.data_lookup[label]] = int(unpack[1])
                # If the label is new, create a spot for it
                else:
                    self.data = self.data + [value]
                    self.data_type = self.data_type + [int(unpack[1])]
                    self.data_lookup[label] = len(self.data)-1
            # Reply to the client that a write was performed
            packet_resp = "$1%"
        else:
            # If reading, then for each requested label
            packet_resp = "&,"
            for label in request:
                # If the label is present in the datastore
                if label in self.data_lookup:
                    # Get the value and type and add them to the reply format
                    val = str(self.data[self.data_lookup[label]])
                    type_ = str(self.data_type[self.data_lookup[label]])
                    packet_resp = packet_resp + val + ":" + type_ + ","
                else:
                    # If the label is not in the store, append an 'N' typed response
                    packet_resp = packet_resp + "0:N,"
            packet_resp = packet_resp + "%"

        return packet_resp

###
#Datatstore server test
#
#  Creates a datastore server, and a client to write to it
#  changing data printouts allow observation of function in
#  real time
###

if __name__ == '__main__':
    data_serv = datashare_server()
    data_serv.start()

    client = datashare_client()


