# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"

        # During each iteration, each state is checked
        # Should a state be a terminal state during the given iteration, we
        # find the highest valued action and use it to find the q Value of the action
        iterationCounter = 0
        while iterationCounter < self.iterations:
            stateValues = util.Counter()
            for currState in self.mdp.getStates():
                if self.mdp.isTerminal(currState) == False:
                    currAction = self.computeActionFromValues(currState)      
                    stateValues[currState] = self.computeQValueFromValues(currState, currAction)
                else:
                    stateValues[currState] = 0
            # Do this in a second loop to allow values to actually update
            for currState in self.mdp.getStates():    
                self.values[currState] = stateValues[currState]
            iterationCounter = iterationCounter + 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
    
        # Calculates Q values using Bellman Equations
        totalQVal = 0
        for i in self.mdp.getTransitionStatesAndProbs(state, action):
            currState = i[1]
            nextState = i[0]
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            nextVal = self.values[nextState]
            totalQVal = totalQVal + currState * (reward + discount * nextVal)
        return totalQVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # If the current state is a terminal state, there are no actions to take 
        if self.mdp.isTerminal(state) == True:
            return 'None'
        # Calls the computeQValuesFromValues function for each action
        # And then returns the highest valued action
        values = util.Counter()
        for currAction in self.mdp.getPossibleActions(state):
            values[currAction] = self.computeQValueFromValues(state, currAction)
        return values.argMax()
    


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
