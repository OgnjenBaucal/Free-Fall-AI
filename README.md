# Free Fall AI

## Problem description
  The game consists of a player and clouds, the goal of the game is to not hit the clouds while falling down.
  The player can accelerate towards right and left to change its position. The longer the player survives the
  faster he starts falling increasing the games difficulty. 

  The inputs are the player's x coordinate, horizontal velocity, the distance between the player and the left 
  and right sides of the gap in the clouds. The inputs vary in sizes so they are normalized to a range of [-1, 1].
  The action space is continous, the agent outputs a number from the [-1, 1] interval which describes the acceleration
  being currently applied to the agent's horizontal velocity.

## Solution
  ### NeuroEvolution of Augmenting Topologies
  NeuroEvolution of Augmenting Topologies (NEAT) is a genetic algorithm that evolves neural networks. The algorithm
  changes the structure and the weights of neural networks, starting with a simple neural network with just an input
  and ouput layer. Then adding on new connections, layers and changes to the weights and biases to discover new 
  structures that might be good at solving a given task.

  ### Evolution process
  The NEAT algorithm begins with a population of simple neural networks, each neural network is encoded as a genome.
  The initial neural network has four inputs and one output. The genome represents the network's structure, 
  weights and biases. Then the networks are tested in their environment and each network is evaluated based on
  how well it performs, the fitness. Networks with the highest fitness are selected to pass on their genes to
  the next generation. The algorithm uses a crossover porcess where genetic material is exchanged between two parent
  networks to create offspring. Mutations alter the network's architecture and weights. New nodes and connections
  can be added, removed or modified. The algorithm continues to evolve the population of neural networks through
  generations until it reaches a solution that meets the desired performance.
