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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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

        current_location = newPos
        threat_location = successorGameState.getGhostPositions()
        vulnerable_threat = newScaredTimes
        food_locations = currentGameState.getFood().asList()
        new_food_locations = newFood.asList()

        "Base score for the start of the game."
        score = 0

        "1) This block of code is responsible for assessing threats."

        "Locate the closest threat to pacman by using the manhattan distance heuristic."
        closest_threat = min([manhattanDistance(current_location, ghost) for ghost in threat_location])

        "If a threat is too close to pacman, then we will incentivize pacman to run away from it by striking-down its score. (Search for a safer places, perhaps with food in them...)"
        if closest_threat < 2:
            return float('-inf')

        "When pacman eats a food source, the threats gets vulnerable, incentivizing pacman through positive addition to its score to seek them out and eat them as long as they are vulnerable."
        if vulnerable_threat[0] > 0:
            score += 100

        " 2) This block of code is for assessing food sources."

        "If pacman ate food, then will improve its score thus incentivizing it to search for more. "
        if len(food_locations) > len(new_food_locations):
            score += 100

        "Locate the closest food source to pacman by using the manhattan distance heuristic."
        if len(new_food_locations) > 0:
            closest_food = min([manhattanDistance(current_location, food) for food in new_food_locations])
        else:
            closest_food = 0

        "If the food source location is too far from pacman current location, then disincentive it from going after it."
        score -= closest_food

        return score



def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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

        pacman = 0
        "Get the legal actions for pacman."
        actions = gameState.getLegalActions(pacman)
        dict = {}

        for action in actions:
            "Generate the next action for pacman."
            successor = gameState.generateSuccessor(pacman, action)
            "Save the action and its value in the dictionary."
            dict[action] = self.minValue(successor, self.depth)

        "Return the key (action) with maximum value. "
        return max(dict, key=dict.get)


    def maxValue(self, gameState, depth, pacman = 0):
        "If we have reached an end-game situation or the maximum depth of minimax, then we will return the evaluation."
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(pacman)
        dict_1 = {}

        for action in actions:
            successor = gameState.generateSuccessor(pacman, action)
            dict_1[action] = self.minValue(successor, depth)

        "Check if a key is in the dictionary."
        for key in list(dict_1.keys()):
            if not key in dict_1:
                return self.evaluationFunction(gameState)
        "Return the maximum value in the dictionary."
        return max(dict_1.values())

    def minValue(self, gameState, depth, ghost = 1):

        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return self.evaluationFunction(gameState)
        "Get the legal actions for the ghosts."
        actions = gameState.getLegalActions(ghost)
        dict_2 = {}

        for action in actions:
            "Generate the next action for a ghost."
            successor = gameState.generateSuccessor(ghost, action)
            "If we have only one ghost, then we will return to pacman."
            "Else: If we have couple of ghosts, then we will move to the next ghost."
            "On both accounts, we will save the action and its value in the dictionary."
            if ghost == gameState.getNumAgents() - 1:
                dict_2[action] = self.maxValue(successor, depth - 1)
            else:
                dict_2[action] = self.minValue(successor, depth, ghost + 1)
        "Check if a key is in the dictionary."
        for key in list(dict_2.keys()):
            if not key in dict_2:
                return self.evaluationFunction(gameState)
        "Return the minimum value in the dictionary."
        return min(dict_2.values())



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        "Evaluate the best course of action using the minimax algorithm."
        action = self.alphaBetaPruning(game_status = gameState, depth = self.depth, agent = 0, alpha = float('-inf'), beta = float('inf'))[1]

        return action

    def alphaBetaPruning(self, game_status, depth, agent, alpha, beta):
        "If we have reached an end-game situation or the maximum depth of minimax, then we will return the evaluation."
        if game_status.isWin() or game_status.isLose() or depth <= 0:
            return self.evaluationFunction(game_status), None

        pacman = 0

        if agent == pacman:
            return self.maxValue(game_status, depth, agent, alpha, beta)
        else:
            return self.minValue(game_status, depth, agent, alpha, beta)

    def maxValue(self, game_status, depth, agent, alpha, beta):

        actions = game_status.getLegalActions(agent)
        max_value = float('-inf')
        max_action = None


        for action in actions:
            successor = game_status.generateSuccessor(agent, action)
            value = self.alphaBetaPruning(successor, depth, agent + 1, alpha, beta)[0]

            if value > max_value:
                max_value = value
                max_action = action

            if max_value > beta:
                return max_value, max_action

            alpha = max(alpha, max_value)

        return max_value, max_action


    def minValue(self, game_status, depth, agent, alpha, beta):

        actions = game_status.getLegalActions(agent)
        min_value = float('inf')
        min_action = None

        for action in actions:
            successor = game_status.generateSuccessor(agent, action)
            number_of_ghost = game_status.getNumAgents() - 1

            "If we have only one ghost, then we will return to pacman."
            "Else: If we have couple of ghosts, then we will move to the next ghost."
            if agent == number_of_ghost:
                value = self.alphaBetaPruning(successor, depth - 1, 0, alpha, beta)[0]
            else:
                value = self.alphaBetaPruning(successor, depth, agent + 1, alpha, beta)[0]

            if value < min_value:
                min_value = value
                min_action = action

            if min_value < alpha:
                return min_value, min_action

            beta = min(beta, min_value)

        return min_value, min_action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = self.expectimax(game_status = gameState, depth = self.depth, agent = 0)[1]
        return actions

    def expectimax(self, game_status, depth, agent):

        pacman = 0
        if game_status.isWin() or game_status.isLose() or depth <= 0:
            return self.evaluationFunction(game_status), None

        if agent == pacman:
            return self.maxValue(game_status, depth, agent)
        else:
            return self.expectimaxValue(game_status, depth, agent)

    def maxValue(self, game_status, depth, agent):
        actions = game_status.getLegalActions(agent)

        max_value = float('-inf')
        max_action = None

        for action in actions:
            successor = game_status.generateSuccessor(agent, action)
            value = self.expectimax(successor, depth, agent + 1)[0]

            if value > max_value:
                max_value = value
                max_action = action

        return max_value, max_action

    def expectimaxValue(self, game_status, depth, agent):
        actions = game_status.getLegalActions(agent)

        "This time, depending on Pac-Man's understanding of how the ghost operate,"
        "he will act according to the operation expectancy of their actions."
        probability = (1.0/len(actions))

        exp_value = 0
        exp_action = None

        for action in actions:
            successor = game_status.generateSuccessor(agent, action)
            number_of_ghosts = game_status.getNumAgents() - 1

            "If we have only one ghost, then we will return to pacman."
            "Else: If we have couple of ghosts, then we will move to the next ghost."
            if agent == number_of_ghosts:
                value = probability * self.expectimax(successor, depth - 1, 0)[0]
            else:
                value = probability * self.expectimax(successor, depth, agent + 1)[0]

            "Sum the expectation values."
            exp_value += value

        return exp_value, exp_action



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
