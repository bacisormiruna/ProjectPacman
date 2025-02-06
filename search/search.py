# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from collections import deque

import util
from game import Directions
from typing import List , Tuple
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # dupa pseudocodul din cursul 2
    noduriVizitate = set() #folosim set ca sa evitam ciclurile
    frontiera = Stack()
    frontiera.push((problem.getStartState(), []))

    while not frontiera.isEmpty():
        stare, cale = frontiera.pop()

        if problem.isGoalState(stare):#daca am ajuns unde ne doream returnam calea
            return cale

        #se parcurge graful/labirintul in adancime
        if stare not in noduriVizitate: #daca starea nu e in set, o adaugam si o marcam ca fiind vizitata, si ii tinem minte succesorii
            noduriVizitate.add(stare)
            noduriSuccesoare = problem.getSuccessors(stare)
            for succesor, mutare, cost in noduriSuccesoare: #aici folosim si cost doar pentru a mentine corectitudinea structurii, la dfs nu conteaza costul
                if succesor not in noduriVizitate:
                    frontiera.push((succesor, cale + [mutare]))  #adaugam in stiva, succesorul si calea pana la el
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    #exact aceeasi idee ca si la dfs, doar ca structura de date folosita este alta
    noduriVizitate = set()
    frontiera = Queue()
    frontiera.push((problem.getStartState(), []))

    while not frontiera.isEmpty():
        stare, cale = frontiera.pop()

        if stare in noduriVizitate:
            continue

        noduriVizitate.add(stare)

        if problem.isGoalState(stare): #daca am ajuns unde ne doream returnam calea
            return cale

        succesori = problem.getSuccessors(stare) #se parcurge graful pe nivele
        for successor, mutare, cost in succesori: #nici aici nu avem nevoie de cost, puteam pune _
            if successor not in noduriVizitate:
                frontiera.push((successor, cale + [mutare]))
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]: #calea cea mai ieftina catre un nod
    """Search the node of least total cost first."""
    #o extindere a bfs cu folosirea unei cozi de prioritati si stocarea costului pe parcurs
    #prioritatea o face costulNou calculat
    start = problem.getStartState()
    frontiera = PriorityQueue() #seamana cu bfs dar folosesc un dictionar pentru a tine cont de cost si o coada de prioritati
    frontiera.push((start, [], 0), 0)

    noduriVizitate = {} #dicționar folosit pentru a ține minte cel mai mic cost al drumului către fiecare nod (ca un vector de apariții).

    while not frontiera.isEmpty():
        stare, cale, cost = frontiera.pop() #tupla (stare, cale cost)
        if stare in noduriVizitate and noduriVizitate[stare] <= cost:
            continue

        noduriVizitate[stare] = cost

        if problem.isGoalState(stare): #daca am ajuns in punctul de stop atunci returnam calea de mutări acumulată
            return cale

        for successor, mutare, costCurent in problem.getSuccessors(stare): #Expansiunea nodului
            costNou = cost + costCurent
            if successor not in noduriVizitate or noduriVizitate[successor] > costNou:
                frontiera.push((successor, cale + [mutare], costNou), costNou) # dacă am găsit un drum mai ieftin, îl adăugăm în coada cu prioritate egală cu costNou.
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    # eureistica euclidiana : manhattan
    # inițializează coada cu priorități A* și adaugă nodul de start împreună cu drumul parcurs și costul
    # in plus fata de UCS tine minte cel mai bun cost si drumul parcurs pana in acel moment
    stareStart = problem.getStartState()
    vizitate = {}
    frontiera = PriorityQueue()
    frontiera.push((stareStart, [], 0), heuristic(stareStart, problem))

    while not frontiera.isEmpty():
        stare, cale, costCurent = frontiera.pop()

        if stare in vizitate and vizitate[stare] <= costCurent:
            continue

        vizitate[stare] = costCurent

        if problem.isGoalState(stare): #am ajuns in punctul in care doream
            return cale

        for succesor, mutare, cost in problem.getSuccessors(stare):
            costNou = costCurent + cost #h(n)
            prioritate = costNou + heuristic(succesor, problem) # f(n) = h(n) + g(n), unde g(n) este euristica (admisibilă și consistentă)

            if succesor in vizitate and costNou >= vizitate[succesor]:
                continue
            frontiera.push((succesor, cale + [mutare], costNou), prioritate) # punem succesorul în coada de priorități cu prioritatea calculata mai sus
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
