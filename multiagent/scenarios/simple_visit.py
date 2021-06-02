import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random

class Scenario(BaseScenario):
     def __init__(self):
        super(BaseScenario, self).__init__()

        # read parameters from config file
        with open('config_simple_visit.json') as file:
            self.config = json.load(fp)
        
        set_seed(self.config['seed'])

    def make_world(self):
        # making a world
        world = World()

        # add good agents
        world.agents = [Agent() for i in range(self.config['ngood_agents'])]
        for i, agent in enumerate(world.agents):
            agent.name = 'Good agent %d' % i
            agent.collide = False
            agent.silent = True
        
        # add adversary agents
        world.agents = [Agent() for i in range(self.config['nadvs_agents'])]
        for i, agent in enumerate(world.agents):
            agent.name = 'Adv agent %d' % i
            agent.collide = False
            agent.silent = True

        # add boxes
        world.boxes = [LoadBox() for i in range(self.config['nboxes'])]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
    
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p) 

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2
        
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def set_seed(self,seed):
        random.seed(seed)

