<h3>Core Basics</h3>

This package contains a set of libraries we've assembled over time, containing tools and assets used across a wide range of projects. The libraries within implement several kinds of objects, algorithms, and datastructures that we consider basic building blocks. Some of these are standards, implemented at the basic level for more control or flexibility, or standardization and internal feedback. Others implement optional efficiency improvements or hooks we prefer to have, and some are creations of our own, which we deploy more generally than individual projects.

Contents:
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

   * Linked Lists - A standard linked list, 

   * Array-linked lists - 

   * Queues - 

   * Stacks - 

   * Priority Queues - 

   * Graphs - 

- _Fundamental Algorithms_:

- _Handy Assets_:

- _Neural Networks_:


