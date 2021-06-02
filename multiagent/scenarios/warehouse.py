import numpy as np
import random
import copy
from multiagent.core import World, Agent, Landmark, LoadBox,isNear , calcDistance
from multiagent.scenario import BaseScenario
from memory_profiler import profile
random.seed(42)
import gc

"""TODOs

Items:
    Add three kind of reward
        Global -> Shared by everyone
        Local  -> Corresponding to box Assignment
        Shared -> If I help in someone else's box
    Calculate this reward at the environment reset part for all of the boxes and then just return it for the agents
"""
     
class Scenario(BaseScenario):
    
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(2)]
        #Add boxes
        world.boxes = [LoadBox() for i in range(2)]        
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.warehouse = True
            agent.assignedBoxes.append("box {}".format(i))
        
        # TODO create a new target field inside boxes 
        #! Did not understand this objective
        for i, box in enumerate(world.boxes):
            box.name = 'box %d' % i
            box.agentAssigned = 'agent {}'.format(i)
            world.boxRewards.append(-5.0)
            
        # add landmarks
        world.landmarks = [Landmark() for i in range(2)]
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
            agent.color = np.array([0.0,0.0,0.0])
            
        world.agents[0].color = np.array([1.0,0.0,1.0])
        world.agents[1].color = np.array([0.0,1.0,1.0])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([1.0,0.0,1.0])
        world.landmarks[1].color = np.array([0.0,1.0,1.0])
        #random properties of boxes
        for i, box in enumerate(world.boxes):
            box.rewardAssigned = random.randint(1,5)
            box.color = np.array([1.0,0.0,0.0])
            world.boxRewards[i] = -5.0 
        world.boxes[0].color = np.array([1.0,0.0,1.0])
        world.boxes[1].color = np.array([0.0,1.0,1.0])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.extraBoxesHandled = []
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, box in enumerate(world.boxes):
            box.pickedUp = False
            box.agentHandling = None
            # box.state.p_pos = copy.deepcopy(world.agents[i].state.p_pos)
            box.state.p_pos = np.random.uniform(-1,+1, world.dim_p)            
            box.state.p_vel = np.zeros(world.dim_p)
            box.pickedUp = False
            box.agentHandling = None
        
        #!initializing the box parameters
        for i,box in enumerate(world.boxes):            
            box.farInit(world.landmarks[i])
        gc.collect()
        
    def reward(self, agent, world):
        rewardAgent = 0
        
        localReward = 0
        sharedReward = 0
        globalReward = 0
        
        
        sharedRewardScalingFactor = 0.0
        globalRewardScalingFactor = 0.0

        
        # TODO add a destination in loadbox and use that to calculate reward not landmark
        localReward = 0
        
        #? Handling Boxes that were assigned to this agent
        for box in agent.assignedBoxes:
            ind = int(box.split()[1])          
            localReward += world.boxRewards[ind]
            
  
                
        sharedReward = 0
        # print(agent.name,">EXTRA BOXES HANDLED>",agent.extraBoxesHandled)
        for boxName in agent.extraBoxesHandled:
            boxId = int(boxName.split()[1])
            sharedReward += sharedRewardScalingFactor*(world.boxRewards[boxId])

        #? A global shared reward
        globalReward = (sum(world.boxRewards)/len(world.boxRewards))*(globalRewardScalingFactor)
        
        #? All rewards are in positive system now
        rewardAgent += localReward               
        rewardAgent += sharedReward
        rewardAgent += globalReward
        
        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 1.0:
                return 0
            if x < 1.1:
                return (x - 1.0) * 10
            return min(np.exp(2 * x - 2.2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rewardAgent -= bound(x)
                          
        # print("{}> Self Boxes Reward {} > Shared Boxes Reward {} > Global Reward >{}".format(agent.name,localReward,sharedReward,globalReward))                    
        return rewardAgent
    
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # TODO incorporate the boxes the agent is assigned to, also what are the boxes goals, etc
        # TODO if it is carrying any box, etc                
        
        returnDict = {}
        
        #Handling the communication observations
        commState = []
        
        for otherAgent in world.agents:
            if otherAgent!=agent:
                commState.append(otherAgent.state.c)
        # commState = np.array(commState)
               
        
        #handling boxes
        myboxObservation = []
        otherboxObservation = []
        for entity in world.boxes:
            if entity.name in agent.assignedBoxes:
                myboxObservation.append(np.array([float(entity.pickedUp)]))
                myboxObservation.append(entity.state.p_pos - agent.state.p_pos)
            else:
                otherboxObservation.append(np.array([float(entity.pickedUp)]))
                otherboxObservation.append(entity.state.p_pos - agent.state.p_pos)

        
        myboxlandmarks = []
        otherboxlandmarks = []
        for i, entity in enumerate(world.landmarks):
            if 'box %d' % i in agent.assignedBoxes:
                myboxlandmarks.append(entity.state.p_pos - agent.state.p_pos)
            else:
                otherboxlandmarks.append(entity.state.p_pos - agent.state.p_pos)
        
        
        entity_pos = []
        other_vel = []
        for entity in world.agents:
            if entity!=agent:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                other_vel.append(entity.state.p_vel)
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + myboxObservation + myboxlandmarks + otherboxObservation + otherboxlandmarks + entity_pos + other_vel)
        return obs
        # TODO Fix this after BTP
        #! Max entities assumed to be three times the number of agents   
        #Padding entity positions
        entity_pos = np.array(entity_pos)            
        entityPositions = np.zeros((entity_pos.shape[0]*3*world.maxAgents,entity_pos.shape[1]))        
        entityPositions[:entity_pos.shape[0],:entity_pos.shape[1]] = entity_pos 
        
        
        
        
        #! Assuming max two times boxes as the agents
        try:
            boxDetails = np.zeros((world.maxBoxes*boxObservation.shape[1], boxObservation.shape[1]))
            if len(boxObservation) > 0:
                boxDetails[:boxObservation.shape[0],:boxObservation.shape[1]] = boxObservation
        except:
            import ipdb; ipdb.set_trace()
        # print("###############",np.concatenate((entityPositions.reshape(-1), boxDetails.reshape(-1), np.array([agent.state.p_vel]).reshape(-1))).shape)
        return np.concatenate((entityPositions.reshape(-1), boxDetails.reshape(-1), np.array([agent.state.p_vel]).reshape(-1)))
        # returnDict = {'comm':commObservation,'entities':entityPositions , 'boxes':boxDetails,"agentVelocity":agent.state.p_vel}
                    
        return returnDict

    def done(self, agent, world):
        return False
