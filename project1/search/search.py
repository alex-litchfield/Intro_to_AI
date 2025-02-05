# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    """
    THIS IS THE PSEUDOCODE!!

    fringe = node
    while True:
        if fringe empty:
            return "failure"
        node.removeFront
        if goal-test:
            return node 
        for child-node:
            fringe.insert(child_node)    """
    
    # Initial variables
    visited_nodes = set() # Set of nodes that have already been visited
    fringe = util.Stack() # The fringe, otherwise known as the stack data structure
    directions_list = [] # List of directions which the pacman agent took
    
    # Add first node/direction tuple to the fringe
    fringe.push((problem.getStartState(), directions_list))

    # Infinite loop
    while True:
        # If fringe is empty, end the loop
        if fringe.isEmpty():
            print("failure")
            return current_node_directions

        # Node/direction tuple in the stack is removed and set to current node variables
        current_node_state, current_node_directions = fringe.pop()

        # Check if the current node is the goal state, return if true
        if problem.isGoalState(current_node_state):
            return current_node_directions
        # Check if the current node is in the set of visited nodes
        if current_node_state not in visited_nodes:
            visited_nodes.add(current_node_state) # Add current node to the list of visited nodes
            for i in problem.getSuccessors(current_node_state): # loop through each successor node
                # Creating distinct variables for later use
                successor_node_state = i[0]
                successor_node_directions = i[1]
                # If the successor node state is not already in the visited node set
                if successor_node_state not in visited_nodes:
                    combined_directions = [] # Complete list of directions that will lead to the successor node
                    # For each direction in the current node direction list, add to the combined directions list
                    for j in current_node_directions:
                        combined_directions.append(j)
                    # Appends the entire list of successor node directions as a single item in the combined_directions list
                    combined_directions.append(successor_node_directions)
                    # Put the successor node in stack
                    fringe.push((successor_node_state, combined_directions))
 

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    # Initial variables
    visited_nodes = set() # Set of nodes that have already been visited
    fringe = util.Queue() # The fringe, otherwise known as the queue data structure
    directions_list = [] # List of directions which the pacman agent took

    # Add first node/direction tuple to the fringe
    fringe.push((problem.getStartState(), directions_list))

    # Infinite loop
    while True:
        # If fringe is empty, end the loop
        if fringe.isEmpty():
            print("failure")
            return current_node_directions

        # Node/direction tuple in the queue is removed and set to current node variables
        current_node_state, current_node_directions = fringe.pop()

        # Check if the current node is the goal state, return if true
        if problem.isGoalState(current_node_state):
            return current_node_directions
        # Check if the current node is in the set of visited nodes
        if current_node_state not in visited_nodes:
            visited_nodes.add(current_node_state) # Add current node to the list of visited nodes
            for i in problem.getSuccessors(current_node_state): # loop through each successor node
                # Creating distinct variables for later use
                successor_node_state = i[0]
                #print("SUCCESSOR STATE:", i[0], "ORIGINAL STATE:", current_node_state)
                successor_node_directions = i[1]
                # If the successor node state is not already in the visited node set
                if successor_node_state not in visited_nodes:
                    combined_directions = [] # Complete list of directions that will lead to the successor node
                    # For each direction in the current node direction list, add to the combined directions list
                    for j in current_node_directions:
                        combined_directions.append(j)
                    # Appends the entire list of successor node directions as a single item in the combined_directions list
                    combined_directions.append(successor_node_directions)
                    # Put the successor node in queue
                    fringe.push((successor_node_state, combined_directions))
                #print("CURR STATE:", current_node_state)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # Initial variables
    visited_nodes = set() # Set of nodes that have already been visited
    fringe = util.PriorityQueue() # The fringe, otherwise known as the priority queue data structure
    directions_list = [] # List of directions which the pacman agent took
    cost = 0 # Initial cost is 0

    # Add first node/direction tuple to the fringe
    fringe.push((problem.getStartState(), directions_list, cost), cost)

    # Infinite loop
    while True:
        # If fringe is empty, end the loop
        if fringe.isEmpty():
            print("failure")
            return current_node_directions

        # Node/direction tuple in the priority queue is removed and set to current node variables
        current_node_state, current_node_directions, current_node_cost = fringe.pop()

        # Check if the current node is the goal state, return if true
        if problem.isGoalState(current_node_state):
            return current_node_directions
        # Check if the current node is in the set of visited nodes
        if current_node_state not in visited_nodes:
            visited_nodes.add(current_node_state) # Add current node to the list of visited nodes
            for i in problem.getSuccessors(current_node_state): # loop through each successor node
                # Creating distinct variables for later use
                successor_node_state = i[0]
                successor_node_directions = i[1]
                successor_node_cost = i[2]
                # If the successor node state is not already in the visited node set
                if successor_node_state not in visited_nodes:
                    combined_directions = [] # Complete list of directions that will lead to the successor node
                    # For each direction in the current node direction list, add to the combined directions list
                    for j in current_node_directions:
                        combined_directions.append(j)
                    # Appends the entire list of successor node directions as a single item in the combined_directions list
                    combined_directions.append(successor_node_directions)
                    # Calculate combined cost
                    combined_cost = current_node_cost + successor_node_cost
                    # Put the successor node in priority queue (second cost value is used as priority)
                    fringe.push((successor_node_state, combined_directions, combined_cost), combined_cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
        # Initial variables
    visited_nodes = set() # Set of nodes that have already been visited
    fringe = util.PriorityQueue() # The fringe, otherwise known as the priority queue data structure
    directions_list = [] # List of directions which the pacman agent took
    cost = 0 # Initial cost is 0

    # Add first node/direction tuple to the fringe
    fringe.push((problem.getStartState(), directions_list, cost), cost)

    # Infinite loop
    while True:
        # If fringe is empty, end the loop
        if fringe.isEmpty():
            print("failure")
            return current_node_directions

        # Node/direction tuple in the priority queue is removed and set to current node variables
        current_node_state, current_node_directions, current_node_cost = fringe.pop()

        # Check if the current node is the goal state, return if true
        if problem.isGoalState(current_node_state):
            return current_node_directions
        # Check if the current node is in the set of visited nodes
        if current_node_state not in visited_nodes:
            visited_nodes.add(current_node_state) # Add current node to the list of visited nodes
            for i in problem.getSuccessors(current_node_state): # loop through each successor node
                # Creating distinct variables for later use
                successor_node_state = i[0]
                successor_node_directions = i[1]
                successor_node_cost = i[2]
                # If the successor node state is not already in the visited node set
                if successor_node_state not in visited_nodes:
                    combined_directions = [] # Complete list of directions that will lead to the successor node
                    # For each direction in the current node direction list, add to the combined directions list
                    for j in current_node_directions:
                        combined_directions.append(j)
                    # Appends the entire list of successor node directions as a single item in the combined_directions list
                    combined_directions.append(successor_node_directions)
                    # Calculate combined cost
                    combined_cost = current_node_cost + successor_node_cost
                    # Calculating priority (utilizes the defined "heuristic" from searchAgents.py that is specified in terminal arguments)
                    priority = combined_cost + heuristic(successor_node_state, problem)
                    # Put the successor node in priority queue
                    fringe.push((successor_node_state, combined_directions, combined_cost), priority)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
