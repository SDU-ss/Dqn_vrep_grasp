import os
import numpy as np

class Configs():

    def __init__(self):
        # hyper parameters
        self.EPSILON = 0.5 # prob of exploration
        self.GAMMA = 0.5 #discount factor
        self.LR = 1e-4 #learning rate
        self.MEMORY_CAPACITY = 500000 # max size of experience dataset
        self.Q_NETWORK_ITERATION = 100 # update the target network from the eval network every 100 steps
        self.BATCH_SIZE = 1
        self.EPISODES = 10000 # training episodes 10000

        self.ROTATION_BINS = 8
        self.DIM_ACTIONS = 224 * 224 * self.ROTATION_BINS
        self.DIM_STATES = (224, 224)

        self.ANGLE_CHANNELS = 1#1 channel, but input 8 images at the same time

        self.WORKSPACE_LIMITS = np.asarray([[-0.705, -0.195], [-0.255, 0.255], [0.01,0.3]])  # define workspace limits in robot coordinates => state image space
        self.ACTION_SPACE = self.WORKSPACE_LIMITS#[[-0.7055, -0.1945], [-0.2555, 0.2555], [0.013, 0.3]]

        self.MAX_OBJ_NUM = 8 # the max object number for every runout
        self.OBJ_MESH_DIR =  os.path.abspath('objects/blocks/')#os.path.abspath('objects/blocks/')  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        self.TEXTURE_DIR = os.path.abspath('objects/textures/')  # Directory for desktop texture

        self.MIN_HEIGHTMAP_ARR = 0.45#0.50472337 #np.min(depth_arrmin:   ,max:
        self.MAX_HEIGHTMAP_ARR = 0.95#0.9083122
        
        





