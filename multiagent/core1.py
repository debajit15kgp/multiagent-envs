import numpy as np
import copy
from memory_profiler import profile

# physical/external base state of all entites

def isNear(box,landmark,threshold=0.05):
    if (np.sum(np.square(box.state.p_pos-landmark.state.p_pos)) <= threshold):
        return True
    else:
        return False

def calcDistance(entity1,entity2):
    return np.sqrt(np.sum(np.square(entity1.state.p_pos - entity2.state.p_pos)))

class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None
        # pickup Action
        self.pickup = False
        # drop Action
        self.drop = False

class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# Properties of the load that we wish to carry
class LoadBox(Entity):
    """
    The load boxes have to reach the corresponding landmarks
    """
    def __init__(self):
        super(LoadBox, self).__init__()
        # Agent which will get reward once this task is done
        self.agentAssigned = None
        # Reward agent will get once the Box Reaches its target
        self.rewardAssigned = None
        # Determines whether any agent has picked up this box
        self.pickedUp = False
        # Agent which is handling this box
        self.agentHandling = None
        #Boxes are pickup but but do not collide
        self.collide = False
        # Distance of te
        self.goalDistInit = None
        #To store the previous state of goalDist of the Box
        self.prevGoalDist = None
        
    def farInit(self,landmark):
        self.goalDistInit = calcDistance(self,landmark)
        self.prevGoalDist = copy.deepcopy(self.goalDistInit)
 
    def rewardDist(self,landmark,rewardMultiplier=2.0,negativeRewardMultiplier=-2.0,stagnantReward = -5.0, nearRewardConstant = 10.0):

        threshold = 0.05
        
        boxReached = isNear(self,landmark,threshold=threshold)        
        
        #! Heavily penalizing taking the box away
        distancePrev = copy.deepcopy(self.prevGoalDist)
        distanceNow = calcDistance(self,landmark)
        
        #? Updating the goal distance in the memory
        self.prevGoalDist = distanceNow
        
        #! Rewarding negative if the box is stagnant
        if distanceNow == distancePrev and not boxReached:
            return stagnantReward
        
        elif distanceNow == distancePrev and boxReached:
            return nearRewardConstant
        
        #! Rewarding negative if the box has been moved away from the target
        # elif distanceNow > distancePrev:
        #     # print("Box Moved Away")
        #     return negativeRewardMultiplier*((distanceNow-distancePrev)/self.goalDistInit)
                
        #How much is the box nearer to the goal compared to where it was initially
        else:
            return rewardMultiplier*(1.0-(distanceNow/self.goalDistInit))
    
    

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        #Names of Assigned Boxes
        self.assignedBoxes = []
        #Stores all of the boxes that the agent has handled on his way
        self.extraBoxesHandled = []
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # is agent from a warehouse task
        self.warehouse = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        # list of all of the landmarks initialized
        self.landmarks = []
        self.walls = []
        #To add the boxes that needs to be transported
        self.boxes = []
        #Reward Associated with the boxes
        self.boxRewards = []
        #Max number of agents expected
        self.maxAgents = 5
        #Max number of boxes expected
        self.maxBoxes = 5
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.boxes + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    #! We have not been using this distance which is being calculated every step
    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    
    def boxRewardCalc(self):
        for i, box in enumerate(self.boxes):
            self.boxRewards[i] = box.rewardDist(self.landmarks[i])                        
    
    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # update boxes states
        self.agent_pick_drop()
        # Calculate all of the rewards
        self.boxRewardCalc()
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()

    def agent_pick_drop(self):
        threshold = 0.25
        busy_agents = []
        # Caculating agents which are busy in carrying boxes
        for box in self.boxes:
            if box.agentHandling:
                busy_agents.append(box.agentHandling)
        # Dropping the boxes and removing the respective agents from busy_list
        for i, agent in enumerate(self.agents):
            # TODO if the agent is stationary only then he can put down the box
            if agent.action.drop:
                for box in self.boxes:
                    if box.agentHandling == agent.name: # and (np.linalg.norm(agent.state.p_vel) < 0.5):
                        agent.color = agent.color + np.ones(agent.color.size)*.5
                        busy_agents.remove(box.agentHandling)
                        box.movable = False
                        box.pickedUp = False
                        box.agentHandling = None
                        box.state.p_pos = copy.deepcopy(agent.state.p_pos)
                        box.state.p_vel = np.zeros(self.dim_p)
                        break
        # TODO better assignment needs to be done (Agent will pick up the box which is most close to him, meanwhile box will be picked up by the agent which is closest)
        for box in self.boxes:
            closest_agent = []
            for agent in self.agents:
                # If agent already has one box with him, he can't pickup another one
                if agent.name in busy_agents: # or (np.linalg.norm(agent.state.p_vel) > 0.5):
                    continue
                # If agent is in threshold distance he is in competition to pickup the box
                # TODO if agent is stationary only then he can pick up the box
                if agent.action.pickup and calcDistance(agent,box) <= threshold:
                    dist = calcDistance(agent,box)
                    closest_agent.append((dist, agent))
            # Select the closest among all agents in threshold distance
            if len(closest_agent) > 0:
                closest_agent.sort()
                #! changing the color of the agent with the box on him
                closest_agent[0][1].color = closest_agent[0][1].color - np.ones(agent.color.size)*.5
                busy_agents.append(closest_agent[0][1].name)

                #? If this box is an extra work that the agent is doing then
                if box.name not in closest_agent[0][1].assignedBoxes and box.name not in closest_agent[0][1].extraBoxesHandled:
                    # print("{} picked extra {}".format(closest_agent[0][1].name,box.name))
                    closest_agent[0][1].extraBoxesHandled.append(box.name)

                box.movable = True
                box.pickedUp = True
                box.agentHandling = closest_agent[0][1].name
                box.state.p_pos = copy.deepcopy(closest_agent[0][1].state.p_pos)  ## ISSUE of assignment(temporary fix using deepcopy)
                box.state.p_vel = copy.deepcopy(closest_agent[0][1].state.p_vel)  ## ISSUE of assignment(temporary fix using deepcopy)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise                
        # apply forces to the pickedup boxes
        for i,box in enumerate(self.boxes):
            if box.pickedUp:
                assert box.agentHandling != None
                agent_id = int(box.agentHandling.split(' ')[1])
                total_force = p_force[agent_id]
                p_force[agent_id] = total_force * (self.agents[agent_id].mass / (box.mass + self.agents[agent_id].mass))
                p_force[i + len(self.agents)] = total_force * (box.mass / (box.mass + self.agents[agent_id].mass))
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf       
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        # Compute pairs of boxes and agents 
        pairs = {}
        for i,box in enumerate(self.boxes):
            if box.agentHandling:
                agent_id = int(box.agentHandling.split(' ')[1])
                pairs[agent_id] = len(self.agents) + i
                pairs[len(self.agents) + i] = agent_id
        # Update the state by taking damping anf forces applied in account
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if i in pairs.keys():
                if (p_force[i] is not None):
                    if (p_force[pairs[i]] is not None):
                        net_force = p_force[i] + p_force[pairs[i]]
                    else:
                        net_force = p_force[i]
                else:
                    if (p_force[pairs[i]] is not None):
                        net_force = p_force[pairs[i]]
                    else:
                        net_force = 0
                net_mass = entity.mass + self.entities[pairs[i]].mass
                entity.state.p_vel += (net_force / net_mass) * self.dt
            else:
                if (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            # set max allowed speed to minimum of max allowed speed of box and agent
            if i in pairs.keys():
                if self.entities[pairs[i]].max_speed is not None:
                    if entity.max_speed is not None:
                        max_speed = min(self.entities[pairs[i]].max_speed, entity.max_speed)
                    else:    
                        max_speed = self.entities[pairs[i]].max_speed
                else:
                    max_speed = entity.max_speed
            else:
                max_speed = entity.max_speed
            if max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)#! Message state is directly set hmmmm if silent
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None] # neither entity moves
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if isinstance(entity_a, Agent) and isinstance(entity_b, LoadBox):
            if entity_b.pickedUp and (entity_b.agentHandling == entity_a.name):
                return [None, None] # agent don't collide with object if it is carrying it
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
            ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force