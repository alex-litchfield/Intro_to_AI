# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        foodList = currentGameState.getFood().asList()
        newCapsuleList = successorGameState.getCapsules()

        # Incentivize NOT moving in the same spot as a ghost
        # Incentivize being FARTHER from a ghost
        largestDistanceToGhost = 0
        for currGhost in newGhostStates:
            # If the move would result in pacman dying to a ghost, return a low score
            if currGhost.getPosition() == newPos:
                # Returns 2 times the negative of the board area, which ideally should not be reached
                # Under normal circumstances other than colliding with a ghost
                return newFood.width * newFood.height * -2
            largestDistanceToGhost = max(manhattanDistance(currGhost.getPosition(), newPos), largestDistanceToGhost)

        # Incentivize being closer to food
        InitializeCheck = True
        smallestDistanceToFood = 0
        # For the first pellet being checked, smallestDistanceToFood will always be set to its manhattan distance from pacman
        # Otherwise, we compare the smallestDistanceToFood with the manhattan distance, keeping the minimum
        for pellet in foodList:
            if InitializeCheck == True or manhattanDistance(pellet, newPos) < smallestDistanceToFood:
                InitializeCheck = False
                smallestDistanceToFood = manhattanDistance(pellet, newPos)
        
        #Incentivizing being closer to capsules
        InitializeCheck = True
        smallestDistanceToCapsule = 0
        # For the first capsule being checked, smallestDistanceToCapsule will always be set to its manhattan distance from pacman
        # Otherwise, we compare the smallestDistanceToCapsule with the manhattan distance, keeping the minimum
        for capsule in newCapsuleList:
            if InitializeCheck == True or manhattanDistance(capsule, newPos) < smallestDistanceToCapsule:
                InitializeCheck = False
                smallestDistanceToCapsule = manhattanDistance(capsule, newPos)

        # Incentivize there being less food on the next board (eating food)
        foodEaten = 0
        # Food was eaten
        if len(foodList) < len(newFood.asList()):
            foodEaten = 10

        # RANKING OF PRIORITY FROM HIGHEST TO LOWEST:
        # Incentivize NOT moving in the same spot as a ghost (die = lose)
        # Incentivize being closer to food (taking optimal path to food is less time, less time is higher score and closer to win)
        # Incentivize being FARTHER from a ghost (more room to work with in a close call)
        # Incentivize being closer to a capsule (a "get out of jail free" card for when you are trapped, and increases score from eating/kills)
        # Incentivize there being less food on the next board (less food on the board which brings us closer to win)

        # Score is always positive (unless colliding with a ghost)
        return 1 / (1 + (largestDistanceToGhost * 0.5) + (smallestDistanceToFood * 4) + (smallestDistanceToCapsule * 2) + foodEaten)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # MaxValue is called because we want to determine the action with the max value
        return self.maxValue(gameState, self.depth, self.index)

    def value(self, currState, currDepth, agentIndex):
        # Returns evalutationFunction of the given state when it is terminal
        if currState.isWin() or currState.isLose() or currDepth == 0:
            return self.evaluationFunction(currState)
        # Returns the max value of the given state when pacman is the current agent
        if agentIndex == 0:
            return self.maxValue(currState, currDepth, agentIndex)
        # Returns the min value of the given state when a ghost is the current agent 
        else:
            return self.minValue(currState, currDepth, agentIndex)

    def maxValue(self, currState, currDepth, agentIndex):
        # Pseudo-initialize v = positive infinity by declaring it as 0 coupled with a check boolean
        # Optimal action holds the best possible action to return in getAction() for an agent to take
        intializeCheck, v, optimalAction = True, 0, None
        # Loops through each successor of the given state
        for currAction in currState.getLegalActions(agentIndex):
            currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1)
            # Determines the maximum between v and the value of the successor
            # Alternatively, should the check boolean be True, v is overriden by the value of the successor
            if intializeCheck == True or currValue > v:
                intializeCheck, optimalAction, v = False, currAction, currValue
        # When the depth has reached the same as self.depth, we return the action
        # Since that is what the end result for getAction() must be
        if currDepth == self.depth:
            return optimalAction
        # Otherwise, we return v, since v is not an action and can help us
        # Determine future actions
        else:
            return v

    def minValue(self, currState, currDepth, agentIndex):
        # Pseudo-initialize v = positive infinity by declaring it as 0 coupled with a check boolean
        intializeCheck, v = True, 0
        # Loops through each successor of the given state
        for currAction in currState.getLegalActions(agentIndex):
            # currValue has a None placeholder value, will be replaced prior to later usage
            currValue = None
            # Assigns a real value to currValue
            if agentIndex < currState.getNumAgents() - 1:
                currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1)
            else:
                currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth - 1, 0)
            # Determines the minimum between v and the value of the successor
            # Should the check boolean be True, v is overriden by the value of the successor
            if intializeCheck == True:
                intializeCheck, v = False, currValue
            # If the check boolean is False, we take the minimum of v and the value of the successor
            else:
                v = min(v, currValue)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, self.depth, self.index, float("-inf"), float("inf"))

    def value(self, currState, currDepth, agentIndex, alpha, beta):
        # Returns evalutationFunction of the given state when it is terminal
        if currState.isWin() or currState.isLose() or currDepth == 0:
            return self.evaluationFunction(currState)
        # Returns the max value of the given state when pacman is the current agent
        if agentIndex == 0:
            return self.maxValue(currState, currDepth, agentIndex, alpha, beta)
        # Returns the min value of the given state when a ghost is the current agent 
        else:
            return self.minValue(currState, currDepth, agentIndex, alpha, beta)

    def maxValue(self, currState, currDepth, agentIndex, alpha, beta):
        # Pseudo-initialize v = positive infinity by declaring it as 0 coupled with a check boolean
        # Optimal action holds the best possible action to return in getAction() for an agent to take
        intializeCheck, v, optimalAction = True, 0, None
        # Loops through each successor of the given state
        for currAction in currState.getLegalActions(agentIndex):
            currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1, alpha, beta)
            # Determines the maximum between v and the value of the successor
            # Alternatively, should the check boolean be True, v is overriden by the value of the successor
            if intializeCheck == True or currValue > v:
                intializeCheck, optimalAction, v = False, currAction, currValue
            # Updates v and alpha values based on alpha beta implementation
            # Does not update when equivelant, more efficient
            if v > beta:
                return v
            alpha = max(alpha, v)
        # When the depth has reached the same as self.depth, we return the action
        # Since that is what the end result for getAction() must be
        if currDepth == self.depth:
            return optimalAction
        # Otherwise, we return v, since v is not an action and can help us
        # Determine future actions
        else:
            return v

    def minValue(self, currState, currDepth, agentIndex, alpha, beta):
        # Pseudo-initialize v = positive infinity by declaring it as 0 coupled with a check boolean
        intializeCheck, v = True, 0
        # Loops through each successor of the given state
        for currAction in currState.getLegalActions(agentIndex):
            # currValue has a None placeholder value, will be replaced prior to later usage
            currValue = None
            # Assigns a real value to currValue
            if agentIndex < currState.getNumAgents() - 1:
                currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1, alpha, beta)
            else:
                currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth - 1, 0, alpha, beta)
            # Determines the minimum between v and the value of the successor
            # Should the check boolean be True, v is overriden by the value of the successor
            if intializeCheck == True:
                intializeCheck, v = False, currValue
            # If the check boolean is False, we take the minimum of v and the value of the successor
            else:
                v = min(v, currValue)
            # Updates v and beta values based on alpha beta implementation
            # Does not update when equivelant, more efficient
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # MaxValue is called because we want to determine the action with the max value
        return self.maxValue(gameState, self.depth, self.index)
    
    def value(self, currState, currDepth, agentIndex):
        # Returns evalutationFunction of the given state when it is terminal
        if currState.isWin() or currState.isLose() or currDepth == 0:
            return self.evaluationFunction(currState)
        # Returns the max value of the given state when pacman is the current agent
        if agentIndex == 0:
            return self.maxValue(currState, currDepth, agentIndex)
        # Returns the expectimax value of the given state when a ghost is the current agent 
        else:
            return self.expValue(currState, currDepth, agentIndex)
    
    def maxValue(self, currState, currDepth, agentIndex):
        # Pseudo-initialize v = positive infinity by declaring it as 0 coupled with a check boolean
        # Optimal action holds the best possible action to return in getAction() for an agent to take
        intializeCheck, v, optimalAction = True, 0, None
        # Loops through each successor of the given state
        for currAction in currState.getLegalActions(agentIndex):
            currValue = self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1)
            # Determines the maximum between v and the value of the successor
            # Alternatively, should the check boolean be True, v is overriden by the value of the successor
            if intializeCheck == True or currValue > v:
                intializeCheck, optimalAction, v = False, currAction, currValue
        # When the depth has reached the same as self.depth, we return the action
        # Since that is what the end result for getAction() must be
        if currDepth == self.depth:
            return optimalAction
        # Otherwise, we return v, since v is not an action and can help us
        # Determine future actions
        else:
            return v

    def expValue(self, currState, currDepth, agentIndex):
        # Initialize v = 0
        v = 0
        # If there are no available actions, v is set to the evaluation function of the given state
        if len(currState.getLegalActions(agentIndex)) == 0:
            v = self.evaluationFunction(currState)
        # If there are available actions to take:
        else:
            # Initialize p as the weighted probability of the possible actions
            p = 1.0 / len(currState.getLegalActions(agentIndex))
            # Loops through each successor of the given state
            for currAction in currState.getLegalActions(agentIndex):
                # The value of v is incremented to account for each action
                if agentIndex < currState.getNumAgents() - 1:
                    v = v + p * self.value(currState.generateSuccessor(agentIndex, currAction), currDepth, agentIndex + 1)
                else:
                    v = v + p * self.value(currState.generateSuccessor(agentIndex, currAction), currDepth - 1, 0)
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    newCapsuleList = currentGameState.getCapsules()

    # Incentivize NOT moving in the same spot as a ghost
    # Incentivize being FARTHER from a ghost
    largestDistanceToGhost = 0
    for currGhost in newGhostStates:
            # If the move would result in pacman dying to a ghost, return a low score
            if currGhost.getPosition() == newPos:
                # Returns 2 times the negative of the board area, which ideally should not be reached
                # Under normal circumstances other than colliding with a ghost
                return newFood.width * newFood.height * -2
            largestDistanceToGhost = max(manhattanDistance(currGhost.getPosition(), newPos), largestDistanceToGhost)

    # Incentivize being closer to food
    InitializeCheck = True
    smallestDistanceToFood = 0
    # For the first pellet being checked, smallestDistanceToFood will always be set to its manhattan distance from pacman
    # Otherwise, we compare the smallestDistanceToFood with the manhattan distance, keeping the minimum
    for pellet in foodList:
        if InitializeCheck == True or manhattanDistance(pellet, newPos) < smallestDistanceToFood:
            InitializeCheck = False
            smallestDistanceToFood = manhattanDistance(pellet, newPos)
    
    
    #Incentivizing being closer to capsules
    InitializeCheck = True
    smallestDistanceToCapsule = 0
    # For the first capsule being checked, smallestDistanceToCapsule will always be set to its manhattan distance from pacman
    # Otherwise, we compare the smallestDistanceToCapsule with the manhattan distance, keeping the minimum
    for capsule in newCapsuleList:
        if InitializeCheck == True or manhattanDistance(capsule, newPos) < smallestDistanceToCapsule:
            InitializeCheck = False
            smallestDistanceToCapsule = manhattanDistance(capsule, newPos)

    # RANKING OF PRIORITY FROM HIGHEST TO LOWEST:
    # Incentivize NOT moving in the same spot as a ghost (die = lose)
    # Incentivize there being less food on the next board (less food on the board which brings us closer to win)
    # Incentivize being closer to a capsule (a "get out of jail free" card for when you are trapped, and increases score from eating/kills)
    # Incentivize being closer to food (taking optimal path to food is less time, less time is higher score and closer to win)
    # Incentivize being FARTHER from a ghost (more room to work with in a close call)

    # Note to grader: When i did this order for evaluationFunction my order was based on me thinking about playing the game myself
    # For this function, I got 4/5 points initially by using the same return value so I started experimenting with random values to get this for 5/5

    return currentGameState.getScore() - largestDistanceToGhost - smallestDistanceToFood - (smallestDistanceToCapsule * 2) - (newFood.count() * 4)
    
# Abbreviation
better = betterEvaluationFunction
