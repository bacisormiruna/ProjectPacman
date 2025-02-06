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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newTimeScared holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newStateGhost = successorGameState.getGhostStates()
        newTimeScared = [ghostState.scaredTimer for ghostState in newStateGhost]

        #Calculul scorului pentru mâncare
        foodList = newFood.asList()
        distFood = [manhattanDistance(newPos, food) for food in foodList]
        distMinToFood = min(distFood) if distFood else 0
        scoreFood = 1.0 / (distMinToFood + 1) #scoreFood: Cu cât mâncarea este mai aproape, cu atât scorul este mai mare (invers proportional)

        #aici avem calculul scorului pentru fantome
        scoreForGhosts = 0
        for stateGhost, timeScared in zip(newStateGhost, newTimeScared):
            distGhost = manhattanDistance(newPos, stateGhost.getPosition())
            if timeScared <= 0:
                scoreForGhosts -= 1.0 / (distGhost + 1)
            else:
                scoreForGhosts += 1.0 / (distGhost + 1)

        if action == Directions.STOP:
            stopScore = -300
        else:
            stopScore = 0

        return successorGameState.getScore() + scoreFood + scoreForGhosts + stopScore

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
        Returns the minimax actiune from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, actiune):
        Returns the successor game state after an agent takes an actiune

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #folosesc doua functii auxiliare de ajutor
        def maxValue(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(0)
            if not legalActions:
                return self.evaluationFunction(state)

            maxScore = float('-inf')

            for action in legalActions:
                successorState = state.generateSuccessor(0, action)
                score = minValue(successorState, depth, 1)
                if score > maxScore:
                    maxScore = score

            return maxScore

        def minValue(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            scoreMin = float('inf')
            if agentIndex < state.getNumAgents() - 1:
                nextAgent = agentIndex + 1
            else:
                nextAgent = 0

            if nextAgent == 0:
                depthNext = depth + 1
            else:
                depthNext = depth

            for actiune in legalActions:
                successorState = state.generateSuccessor(agentIndex, actiune)
                if nextAgent == 0:
                    score = maxValue(successorState, depthNext)
                else:
                    score = minValue(successorState, depthNext, nextAgent)
                scoreMin = min(scoreMin, score)

            return scoreMin

        def actionScore(action):
            stateSuccesor = gameState.generateSuccessor(0, action)
            return minValue(stateSuccesor, 0, 1)

        legalActions = gameState.getLegalActions(0)

        best = None
        scoreBest = float('-inf')
        for actiune in legalActions:
            score = actionScore(actiune)
            if score > scoreBest:
                scoreBest = score
                best = actiune

        return best


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    #update pentru alpha
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                nextAgent = agentIndex + 1
                if agentIndex == state.getNumAgents() - 1:
                    depth += 1
                    nextAgent = 0
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, depth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    #update pentru beta
                    beta = min(beta, value)
                return value

        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            valoare = alphaBeta(successor, 0, 1, alpha, beta)
            if valoare > alpha:
                alpha = valoare
                bestAction = action

        return bestAction

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


        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
