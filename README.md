<h3>Core Basics</h3>

This package contains a set of libraries we've assembled over time, containing tools and assets used across a wide range of projects. The libraries within implement several kinds of objects, algorithms, and datastructures that we consider basic building blocks. Some of these are standards, implemented at the basic level for more control or flexibility, or standardization and internal feedback. Others implement optional efficiency improvements or hooks we prefer to have, and some are creations of our own, which we deploy more generally than individual projects.

**Contents:**
- _Agent Control_: A suite of agent-based control algorithms which are helpful for rapidly implementing an executive control system for software, particularly avoiding vetting the system iteself and eliminating errors from that side fo development. Implements most utilities with optional parallelized threading. Includes:

    * General Purpose State machine - Probably the most fundamental controller

    * Hierarchical Subsumption - For interruption and priority-based execution systems

    * QLearning - Several models, for learning, or pre-training based execution (loading in a manual Q table for instance), with several menu options for learning and execution

    * Murin - Our temporally modified reinforcement learning algorithm, for behavior-based systems. See that project for more abstract details

- _Core Combinatorics_: A set of our most fundamental and commonly used combinatorics algorithms, finding use wherever planning or optimization is needed. Includes:

  * General purpose Genetic algorithms - For applying to a problem where dynamics are opaque or unknown (or you're feeling lazy!)

  * Longest Processing Time algorithm - A reliable stand-by for shop-scheduling compatible problems
  K-nearest neighbours classifier - A solid low-complexity clustering algorithm, also useful in graph-based and vision algorithms

  * Knapsack Algorithm - Both a sorted value to weight ratio heuristic and the standard 2D dynamic algorithm, for selection and packing problems

  * Kruskal's algorithm - For finding the minimum spanning forest within a graph, useful as a base for all kinds of graph-based combinatorics, founding heuristics especially

- _Data Structures_: A bunch of objects implementing specific data structures, particularly emphasizing building computationally efficient manipulation methods. Includes:

   * Alias Trees - A label-linked tree structure for managing aliased labels, mapping sets reflectively. Makes an O(1) access time to fetch the root label of any from a set of aliases for a data point, and keeps the one to one mapping consistent across additions

   * Sorted Lists - A list which keeps all entries in sorted order perpetually using block searching on insert

   * Linked Lists - A standard linked list, with add and remove wrapped for convenience

   * Union trees - The incredibly simple and elegant, yet wildly powerful set-based tree structure

   * Array-linked lists - A joint Array/Linked list object (hence 'Alley') which maps a linked list with an array-based index to allow O(1) access to link elements without iterating over the list. Uses an insertion array to also manage insertion sort. Reflexive construction makes going from array to linked list properties trivial. More memory hungry because of hash-like indexing mechanism.

   * Queues - A standard queue, wrapped methods for convenience

   * Stacks - A standard stack, wrapped methods for convenience

   * Priority Queues - A briority queue which merges a sorted list and a queue, to make managing ordered pop/push operations fast and direct

   * Graphs - A general purpose graph representation, designed to be broad and flexible. Includes node and edge objects which natively handle directed, undirected, and weighted or non-weighted graphs seamlessly. The graph itself supports object-based, list-based, connectivity array based, and connectivity-pair based modes of structure. Dijkstra's algorithm (our favorite!) is implemented here because handling all those modes in it is easier inside the object. Also includes a multi-format input constructor, so graphs can be loaded in freely in any convenient format.

- _Fundamental Algorithms_: A package containing basic algorithms which are used in many places, including extrnsively throughout the other libraries in this package. They are designed to use many efficiency best-practices, while being transparent and possible to easily add hooks or modify with problem-specific augmentations and optimizations. Includes:

   * Block search - A standard log2 based divide and conquer search on a sorted list, accepting a abstracted 'get' function for constructing a relation operator

   * Quicksort - Everyone's favorite recursive sorting algorithm, implemented to accept an abstracted relation operation.

- _Handy Assets_: A collection of useful miscellaneous objects which often see use in other projects, most of which are geared towards sharing across processes and communication. Includes:

   * Datashare Utility - A local-host based data sharing utility which operates on a publisher/subscriber model to pass data to datashare clients using a compact typed data packet structure. Includes real-time update any cycling, with support for both multi-threaded and polled update options. Also includes an abstracted interface protocol so that either TCP sockets, or serialized ports (notably hardware Serial, I2C, or SPI ports) can be used to incorporate pub/sub members into the network. Backbone of the ROSFuse project.

   * Timer - A basic timer object designed to maintain consistent clocking across processes, or within a process, but independent of it.

   * Abstract Variable - A variable wrapper with common member slots, copy function, access inhibitor, and get/set methods, designed to be a cross-process container. Allows for locking of access in priority operations and semi-mutable access in different threads.

- _Neural Networks_: A set of fundamental neural-network based functions, particularly ones that have been thoroughly vetted for operational functionality, accuracy, efficiency, and accessibility. Designed to support multiple modes such as individual or batch training, built-in optional I/O normalization, ordered or unordered training, bias input, and bias training. Also designed to allow for direct modification with problem-specific modifications. We often use the smaller, less powerful types in combinations, either combining multiple of the same, or heterogenous sets, and training them in parallel, then using a weighted accuracy metric to pick an answer from all of them. We also use them as subprocess solvers in hierarchical systems, where a high-level controller applys a small NN quickly to train up a classifier or behavior from small input sets to, essentially, build up a generic subprocess for a subtask in a hierarchy. Sometimes a very effective method here is training several in parallel, seeking their input, trying one, then if the answer was wrong, trying any of the once that didn't pick the known wrong answer and then batch training all on that information. Includes:

   * Error Back-propagation - The standard hidden-layer network structure, taking input vectors and mapping them through a hidden layer onto an output. Includes all the options above, and extensively vetted for speed, correctness, and stability. 

   * Autoencoders - An implementation of an I-I autoencoder, based on the EBP module with specific adjustments to automate self mapping training at all modes. Particularly useful for data classification, input compression, and pattern extraction.

   * Perceptron - One of the simplest and earliest neural networks. Incredibly weak on its own, but fast, simple, and low overhead, so it makes a great part of a multicomponent system. In particular we have luck training several of them in parallel and applying a weighted-by-accuracy or answer-clustering combination approach. Also makes for a good small-scale single purpose agent solver in larger hierarchical systems.

   * Winner-take-all - A handy network for problems that are well conditioned to it, because it tends to learn fast and is low-overhead like the perceptron. Very little nuance in them, but good for quickly learning simple subproblems, like mentioned in the multi-testing above. They are also good for learning to turn the multi-agent wisdom of crowds outputs into actions- when trained on the input to all the others, _and the outputs of the others_.

   * Restricted Boltzmann Machines - Very fun and interesting, but tending to be finnicky, the RBMs are pretty good auto-encoders, but really shine in the task of doing missing-data interpolation when trained on a representative set and then allowed to fill in missing cells.

   * Hebbian Networks - Another early netowrk, and useful as a lightweight unsupervised layer. Not really suited to a lot of problems, but helpfully covers areas that the perceptron isn't suited to, and with a similar computational footprint.
