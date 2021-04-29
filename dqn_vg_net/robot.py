import socket
import select
import struct
import time
import numpy as np
import utils
import vrep
import random
import cv2
from threading import Timer
import h5py
import pickle

# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input
# from keras.models import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# base_model = ResNet50(weights='imagenet')
# layer_model = Model(inputs=base_model.input, outputs=base_model.layers[175].output)#172

class Robot(object):
    def __init__(self, obj_mesh_dir, num_obj,workspace_limits,is_testing,test_preset_cases, test_preset_file):

        #self.is_sim = is_sim
        self.workspace_limits = workspace_limits

        # If in simulation...
        #if self.is_sim:

        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                           [89.0, 161.0, 79.0], # green
                                           [156, 117, 95], # brown
                                           [242, 142, 43], # orange
                                           [237.0, 201.0, 72.0], # yellow
                                           [186, 176, 172], # gray
                                           [255.0, 87.0, 89.0], # red
                                           [176, 122, 161], # purple
                                           [118, 183, 178], # cyan
                                           [255, 157, 167]])/255.0 #pink

        # Read files in object mesh directory
        self.obj_mesh_dir = obj_mesh_dir  
        self.num_obj = num_obj  
        self.mesh_list = os.listdir(self.obj_mesh_dir)
        #print (self.mesh_list)

        # Randomly choose objects to add to scene
        # self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        #print (self.obj_mesh_ind)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        #set the object file
        self.is_testing = False
        self.test_preset_cases = test_preset_cases
        self.test_preset_file = test_preset_file

        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # If testing, read object meshes and poses from test case file
        if self.is_testing and self.test_preset_cases:
            file = open(self.test_preset_file, 'r')
            file_content = file.readlines()
            self.test_obj_mesh_files = []
            self.test_obj_mesh_colors = []
            self.test_obj_positions = []
            self.test_obj_orientations = []
            for object_idx in range(self.num_obj):
                file_content_curr_object = file_content[object_idx].split()
                self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
                self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
                self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
                self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
            file.close()
            self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)

        # Add objects to simulation environment
        #self.add_objects()

    def setup_sim_camera(self):

        # Get handle to camera
        #sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho', vrep.simx_opmode_blocking) #顶部
        sim_ret, self.cam_handle_oppo = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp',vrep.simx_opmode_blocking) #对面
        sim_ret, self.cam_handle_left = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp2',vrep.simx_opmode_blocking) #左手边
        sim_ret, self.cam_handle_right = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp1',vrep.simx_opmode_blocking) #右手边


        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        #self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img, self.bg_color_img, color_img_oppo, color_img_left, color_img_right = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def init_getObjHandle(self):
        self.object_handles = []
        block_list = ['shape1','shape2','shape3','shape4','shape5','shape6','shape7','shape8']
        for object_idx in range(8):
            sim_ret, temp_handle = vrep.simxGetObjectHandle(self.sim_client, block_list[object_idx],vrep.simx_opmode_blocking)
            self.object_handles.append(temp_handle)

    def add_objects(self,texture_dir):
        # set random texture background
        textureNameList = os.listdir(texture_dir)
        randNameInd = random.randint(0, len(textureNameList) - 1)
        textureFileName = texture_dir + '/' + textureNameList[randNameInd]  # 'objects/texture/texture1.jpg'
        # print("texture.Name:",textureFileName)
        # add texture
        aa, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                        'remoteApiCommandPlane',
                                                                                        vrep.sim_scripttype_childscript,
                                                                                        'createTexture', [], [],
                                                                                        [textureFileName], bytearray(),
                                                                                        vrep.simx_opmode_blocking)
        random_obj_num = self.num_obj - 2#np.random.randint(1,self.num_obj+1)#3
        # print('all num:',self.num_obj,', rand:',random_obj_num)
        obj_name_list = os.listdir(self.obj_mesh_dir)
        obj_item_list = random.sample(obj_name_list, random_obj_num)

        position=[[-0.6,0.1,0.1],[-0.4,0.1,0.1],[-0.2,0.1,0.1],[-0.6,-0.1,0.1],[-0.4,-0.1,0.1],[-0.2,-0.1,0.1]]
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(random_obj_num):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, obj_item_list[object_idx])
            if self.is_testing and self.test_preset_cases:
                curr_mesh_file = self.test_obj_mesh_files[object_idx]
            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            # object_position = [drop_x, drop_y, 0.18]
            object_position = position[object_idx]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            if self.is_testing and self.test_preset_cases:
                object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                #set the rotation of objects
                object_orientation = [0,0,0] #[self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_orientation = [0,0,0]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]

            print('file',curr_mesh_file)
            print (object_position + object_orientation + object_color)
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            try:
                curr_shape_handle = ret_ints[0]
            except:
                return 0
            self.object_handles.append(curr_shape_handle)
            # sleep
            time.sleep(0.3)#0.5
        self.prev_obj_positions = []
        self.obj_positions = []
        time.sleep(0.5)#1.5
        return random_obj_num


    def restart_sim(self):

        sim_ret, self.UR5_joint1_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint1', vrep.simx_opmode_blocking)
        sim_ret, self.UR5_joint2_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint2', vrep.simx_opmode_blocking)
        sim_ret, self.UR5_joint3_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint3', vrep.simx_opmode_blocking)
        sim_ret, self.UR5_joint4_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint4', vrep.simx_opmode_blocking)
        sim_ret, self.UR5_joint5_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint5', vrep.simx_opmode_blocking)
        sim_ret, self.UR5_joint6_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint6', vrep.simx_opmode_blocking)

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached


    # def stop_sim(self):
    #     if self.is_sim:
    #         # Now send some data to V-REP in a non-blocking fashion:
    #         # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

    #         # # Start the simulation
    #         # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # # Stop simulation:
    #         # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    #         vrep.simxGetPingTime(self.sim_client)

    #         # Now close the connection to V-REP:
    #         vrep.simxFinish(self.sim_client)


    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations


    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):

        #if self.is_sim:

        # Get color image from simulation, 4个相机的
        ############ 顶部 #################
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        # ############ 对面 #################
        # sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle_oppo, 0, vrep.simx_opmode_blocking)
        # color_img_oppo = np.asarray(raw_image)
        # color_img_oppo.shape = (resolution[1], resolution[0], 3)
        # color_img_oppo = color_img_oppo.astype(np.float)/255
        # color_img_oppo[color_img_oppo < 0] += 1
        # color_img_oppo *= 255
        # color_img_oppo = np.fliplr(color_img_oppo)
        # color_img_oppo = color_img_oppo.astype(np.uint8)
        # ############ 左边 #################
        # sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle_left, 0, vrep.simx_opmode_blocking)
        # color_img_left = np.asarray(raw_image)
        # color_img_left.shape = (resolution[1], resolution[0], 3)
        # color_img_left = color_img_left.astype(np.float)/255
        # color_img_left[color_img_left < 0] += 1
        # color_img_left *= 255
        # color_img_left = np.fliplr(color_img_left)
        # color_img_left = color_img_left.astype(np.uint8)
        # ############ 右边 #################
        # sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle_right, 0, vrep.simx_opmode_blocking)
        # color_img_right = np.asarray(raw_image)
        # color_img_right.shape = (resolution[1], resolution[0], 3)
        # color_img_right = color_img_right.astype(np.float)/255
        # color_img_right[color_img_right < 0] += 1
        # color_img_right *= 255
        # color_img_right = np.fliplr(color_img_right)
        # color_img_right = color_img_right.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        # else:
        #     # Get color and depth image from ROS service
        #     color_img, depth_img = self.camera.get_data()
        #     # color_img = self.camera.color_data.copy()
        #     # depth_img = self.camera.depth_data.copy()
        color_img_oppo, color_img_left, color_img_right = (0,0,0)

        return depth_img, color_img, color_img_oppo, color_img_left, color_img_right


    def save_state(self,arr_color,arr_depth):
        nums=os.listdir('state_image/')
        id = len(nums)+1
        nums_dep=os.listdir('state_depth/')
        id_dep = len(nums_dep)+1

        cv2.imwrite('state_image/state_'+str(id)+'.png', cv2.cvtColor(arr_color, cv2.COLOR_RGB2BGR))
        np.save('state_depth/state_' + str(id_dep) + '.npy', arr_depth.astype(np.float32))
        return 'state_image/state_'+str(id)+'.png','state_depth/state_'+str(id_dep)+'.npy'

    def save_state_eval(self,arr_color,arr_depth):
        nums=os.listdir('state_image_eval/')
        id = len(nums)+1
        nums_dep=os.listdir('state_depth_eval/')
        id_dep = len(nums_dep)+1

        cv2.imwrite('state_image_eval/state_'+str(id)+'.png', cv2.cvtColor(arr_color, cv2.COLOR_RGB2BGR))
        np.save('state_depth_eval/state_' + str(id_dep) + '.npy', arr_depth.astype(np.float32))
        return 'state_image_eval/state_'+str(id)+'.png','state_depth_eval/state_'+str(id_dep)+'.npy'

    ### 改用npy保存深度信息，保证小数点后7位的精度  ###
    def get_current_state(self):

        # Get color image from simulation, 4个相机的
        ############ 顶部 #################
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        resize_color = cv2.resize(color_img, (224, 224), interpolation=cv2.INTER_AREA)

        # if self.is_sim:
        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        theta_factor = 1.1018
        depth_img = depth_img * (zFar - zNear) + zNear
        resize_depth = cv2.resize(depth_img, (224, 224), interpolation=cv2.INTER_AREA)
        resize_depth = resize_depth * theta_factor  # scale to about 0.9, for the correspondence between sim and real

        ######  save test ########
        # #depth_img = cv2.resize(depth_img, (227, 227), interpolation=cv2.INTER_AREA)
        # f_hd = h5py.File('see_dep.h5', 'w')  # 创建一个h5文件，文件指针是f
        # f_hd['data'] = depth_img  # 将数据写入文件的主键data下面
        # f_hd.close()
        #
        # # Pickle dictionary using protocol 0.
        # output = open('data.pkl', 'wb')
        # pickle.dump(depth_img, output)
        # output.close()
        #
        # np.save('see_depth.npy',depth_img)
        ######  save test #######

        #shrink = cv2.resize(depth_img, (227,227), interpolation=cv2.INTER_AREA)
        img_path,depth_path = self.save_state(resize_color,resize_depth)

        #shrink = cv2.resize(depth_img, (256,256), interpolation=cv2.INTER_AREA)

        ##########  extract feature  ############
        # # img_path = 'observation1.jpg'
        # img = image.load_img(img_path, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # features = layer_model.predict(x)
        #print('feature.shape:',features.shape)
        # features = []

        return depth_path

    def get_current_state_eval(self):

        # Get color image from simulation, 4个相机的
        ############ 顶部 #################
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        resize_color = cv2.resize(color_img, (224, 224), interpolation=cv2.INTER_AREA)

        # if self.is_sim:
        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        theta_factor = 1.1018
        depth_img = depth_img * (zFar - zNear) + zNear
        resize_depth = cv2.resize(depth_img, (224, 224), interpolation=cv2.INTER_AREA)
        resize_depth = resize_depth * theta_factor  # scale to about 0.9, for the correspondence between sim and real

        ######  save test ########
        # #depth_img = cv2.resize(depth_img, (227, 227), interpolation=cv2.INTER_AREA)
        # f_hd = h5py.File('see_dep.h5', 'w')  # 创建一个h5文件，文件指针是f
        # f_hd['data'] = depth_img  # 将数据写入文件的主键data下面
        # f_hd.close()
        #
        # # Pickle dictionary using protocol 0.
        # output = open('data.pkl', 'wb')
        # pickle.dump(depth_img, output)
        # output.close()
        #
        # np.save('see_depth.npy',depth_img)
        ######  save test #######

        #shrink = cv2.resize(depth_img, (227,227), interpolation=cv2.INTER_AREA)
        img_path,depth_path = self.save_state_eval(resize_color,resize_depth)

        #shrink = cv2.resize(depth_img, (256,256), interpolation=cv2.INTER_AREA)

        ##########  extract feature  ############
        # # img_path = 'observation1.jpg'
        # img = image.load_img(img_path, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # features = layer_model.predict(x)
        #print('feature.shape:',features.shape)
        # features = []

        return depth_path

    def parse_tcp_state_data(self, state_data, subpackage):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        robot_message_type = data_bytes[4]
        assert(robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data' : 1, 'cartesian_info' : 4, 'force_mode_data' : 7, 'tool_data' : 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0,0,0,0,0,0]
            target_joint_positions = [0,0,0,0,0,0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0,0,0,0,0,0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        assert(data_length == 812)
        byte_idx = 4 + 8 + 8*48 + 24 + 120
        TCP_forces = [0,0,0,0,0,0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8

        return TCP_forces


    def close_gripper(self, async=False):


        ######  left and right ##########
        sim_ret, RG2_gripper_left_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_leftJoint0',vrep.simx_opmode_blocking)
        # sim_ret, RG2_gripper_right_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_rightJoint0',vrep.simx_opmode_blocking)
        # print('handle:',RG2_gripper_left_handle)


        #if self.is_sim:
        gripper_motor_velocity = -0.5 # -0.5
        gripper_motor_force = 100#80
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.047: # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            # print(gripper_joint_position)
            if new_gripper_joint_position >= gripper_joint_position:

                # sim_ret, gripper_joint_left_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_left_handle,vrep.simx_opmode_blocking)
                # self.record_force_when_close_gripper()
                # l_f, r_f = self.return_force_when_close_gripper()
                # print('margin: ',self.cal_margin_lefttouch_to_righttouch())
                return self.cal_margin_lefttouch_to_righttouch()
            gripper_joint_position = new_gripper_joint_position
            #print ("close: ", gripper_joint_position)
        gripper_fully_closed = True
        # sim_ret, gripper_joint_left_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_left_handle,
        #                                                                  vrep.simx_opmode_blocking)


        # ## cal left and right gripper pos
        # sim_ret, RG2_gripper_handle_leftTouch = vrep.simxGetObjectHandle(self.sim_client, 'RG2_leftJoint1', vrep.simx_opmode_blocking)
        # sim_ret, RG2_gripper_handle_rightTouch = vrep.simxGetObjectHandle(self.sim_client, 'RG2_rightJoint1',
        #                                                                  vrep.simx_opmode_blocking)
        # sim_ret, gripper_joint_position_leftTouch = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle_leftTouch, vrep.simx_opmode_blocking)
        # sim_ret, gripper_joint_position_rightTouch = vrep.simxGetJointPosition(self.sim_client,
        #                                                                        RG2_gripper_handle_rightTouch,
        #                                                                       vrep.simx_opmode_blocking)
        # #print('close:',gripper_joint_position_leftTouch,gripper_joint_position_rightTouch)
        # sensorName_left = 'RG2_leftForceSensor'
        # sensorName_right = 'RG2_rightForceSensor'
        # sim_ret, RG2_gripper_handle_leftsensor = vrep.simxGetObjectHandle(self.sim_client, sensorName_left,
        #                                                                  vrep.simx_opmode_blocking)
        # sim_ret, RG2_gripper_handle_rightsensor = vrep.simxGetObjectHandle(self.sim_client, sensorName_right,
        #                                                                   vrep.simx_opmode_blocking)
        # # sim_ret, gripper_joint_position_leftTouch = vrep.simxGetJointPosition(self.sim_client,
        # #                                                                       RG2_gripper_handle_leftsensor,
        # #                                                                       vrep.simx_opmode_blocking)
        # # sim_ret, gripper_joint_position_rightTouch = vrep.simxGetJointPosition(self.sim_client,
        # #                                                                        RG2_gripper_handle_rightsensor,
        # #                                                                       vrep.simx_opmode_blocking)
        # sim_ret, gripper_joint_position_leftTouch = vrep.simxGetObjectPosition(self.sim_client, RG2_gripper_handle_leftsensor, -1, vrep.simx_opmode_blocking)
        # print('close pos:',gripper_joint_position_leftTouch)
        # ## ------------------------------

        return self.cal_margin_lefttouch_to_righttouch()

        #return gripper_fully_closed,gripper_joint_left_position

    def open_gripper(self, async=False):

        start_t = time.clock()
        is_blocked = False
        #if self.is_sim:
        gripper_motor_velocity = 0.5 #0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        while gripper_joint_position < 0.0321: # Block until gripper is fully open
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_oneshot)

            elapsed_t = (time.clock() - start_t)
            if elapsed_t >= 5:#block more than 3 seconds, kill
                is_blocked = True
                break
            #print ("open: ", gripper_joint_position)

        block_centers, block_angles = self.get_obj_positions_and_orientations()
        for point_item in block_centers:
            if point_item[2] > 0.065:
                is_blocked = True
                break

        # ## cal left and right gripper pos
        # sim_ret, RG2_gripper_handle_leftTouch = vrep.simxGetObjectHandle(self.sim_client, 'RG2_leftJoint1',
        #                                                                  vrep.simx_opmode_blocking)
        # sim_ret, RG2_gripper_handle_rightTouch = vrep.simxGetObjectHandle(self.sim_client, 'RG2_rightJoint1',
        #                                                                   vrep.simx_opmode_blocking)
        # sim_ret, gripper_joint_position_leftTouch = vrep.simxGetJointPosition(self.sim_client,
        #                                                                       RG2_gripper_handle_leftTouch,
        #                                                                       vrep.simx_opmode_blocking)
        # sim_ret, gripper_joint_position_rightTouch = vrep.simxGetJointPosition(self.sim_client,
        #                                                                        RG2_gripper_handle_rightTouch,
        #                                                                        vrep.simx_opmode_blocking)
        # # print('open:', gripper_joint_position_leftTouch, gripper_joint_position_rightTouch)
        # ## ------------------------------

        return is_blocked

    def close_gripper_robotiq(self, async=False):
        vrep.simxSetStringSignal(self.sim_client, 'robotiq85', 'true', vrep.simx_opmode_oneshot)

        gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper_robotiq(self, async=False):
        vrep.simxSetStringSignal(self.sim_client, 'robotiq85', 'false', vrep.simx_opmode_oneshot)

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data


    def move_to(self, tool_position, tool_orientation):

        #if self.is_sim:

        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02*move_direction/move_magnitude #0.02
        num_move_steps = int(np.floor(move_magnitude/0.02))#0.02

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)

    def execute_v_single(self,Vel,jointNo):

        returnCode = -99999
        if jointNo == 1 :
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint1_handle, Vel , vrep.simx_opmode_blocking)
        elif jointNo == 2 :
            #ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint2_handle,vrep.simx_opmode_blocking)
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint2_handle, Vel, vrep.simx_opmode_blocking)
        elif jointNo == 3 :
            #ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint3_handle,vrep.simx_opmode_blocking)
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint3_handle, Vel, vrep.simx_opmode_blocking)
        elif jointNo == 4 :
            #ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint4_handle,vrep.simx_opmode_blocking)
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint4_handle, Vel, vrep.simx_opmode_blocking)
        elif jointNo == 5 :
            #ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint5_handle,vrep.simx_opmode_blocking)
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint5_handle, Vel, vrep.simx_opmode_blocking)
        else :
            #ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint6_handle,vrep.simx_opmode_blocking)
            returnCode = vrep.simxSetJointTargetVelocity(self.sim_client, self.UR5_joint6_handle, Vel, vrep.simx_opmode_blocking)

        return returnCode
        #simxSetJointTargetVelocity(self.sim_client, self.UR5_joint2_handle, 2, vrep.simx_opmode_blocking)
        #simxSetJointTargetVelocity(self.sim_client, self.UR5_joint3_handle, 2, vrep.simx_opmode_blocking)
        #simxSetJointTargetVelocity(self.sim_client, self.UR5_joint4_handle, 2, vrep.simx_opmode_blocking)
        #simxSetJointTargetVelocity(self.sim_client, self.UR5_joint5_handle, 2, vrep.simx_opmode_blocking)
        #simxSetJointTargetVelocity(self.sim_client, self.UR5_joint6_handle, 2, vrep.simx_opmode_blocking)

    def getVel(self,jointNo):
        arr1=[]
        arr2=[]
        if jointNo == 1 :
            ret,arr1,arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint1_handle,vrep.simx_opmode_blocking)
        elif jointNo == 2 :
            ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint2_handle,vrep.simx_opmode_blocking)
        elif jointNo == 3 :
            ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint3_handle,vrep.simx_opmode_blocking)
        elif jointNo == 4 :
            ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint4_handle,vrep.simx_opmode_blocking)
        elif jointNo == 5 :
            ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint5_handle,vrep.simx_opmode_blocking)
        else :
            ret, arr1, arr2 = vrep.simxGetObjectVelocity(self.sim_client,self.UR5_joint6_handle,vrep.simx_opmode_blocking)

        return arr1,arr2

    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1 # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01*increment/np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0],increment_position[1],increment_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2 # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success


    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        self.tcp_socket.close()


    def go_home(self):

        self.move_joints(self.home_joint_config)


    # Note: must be preceded by close_gripper()
    def check_grasp(self):

        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26


    # Primitives ----------------------------------------------------------

    '''
    action = (p_x,p_y,p_z,theta) # x,y is the output prediction, z is estimated using corresponding depth information
    '''
    def step(self,action_2):# (output_action_type(0~36, step 20), output_action[0],output_action[1],output_action[2],output_action[3])

        is_blocked = self.open_gripper()
        if is_blocked:
            return -1, -1

        reward = 0
        workspace_limits = self.workspace_limits#np.asarray([[-0.7055, -0.1945], [-0.2555, 0.2555], [0.013,0.3]])
        position = (action_2[0],action_2[1],action_2[2])
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (action_2[3] % np.pi) - np.pi / 2
        # Avoid collision with floor
        position = np.asarray(position).copy()
        #position[2] = max(position[2], workspace_limits[2][0] + 0.02)  # avoid touch floor

        # Move gripper to location above grasp target
        location_above_grasp_target = (position[0], position[1], position[2])

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
            UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
            UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
            UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
            np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)

            # time.sleep(0.05)#0.05

        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

        # Close gripper to grasp target
        g_margin1 = self.close_gripper()
        if g_margin1 > 0.01:
            self.pick_and_hold(action_2)
            time.sleep(1)
            g_margin2 = self.close_gripper()
            if g_margin2 > 0.01:
                reward = 1
                finish = 1
                self.move_grasped_object_to_conner()
            else:
                reward = 0
                finish = 0
                self.go_to_reset_position()
        else:
            # self.pick_and_hold(action_2)
            self.go_to_reset_position()
            reward = 0
            finish = 0

        return reward,finish

    def go_to_reset_position(self):
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        position = (-0.2, -0.125, 0.3)

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (0 % np.pi) - np.pi / 2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        #position[2] = max(position[2], workspace_limits[2][0] + 0.02)  # -0.04

        # Move gripper to location above grasp target
        # grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2])

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)

            # time.sleep(0.05)#0.05

        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

    def push_action(self,current_pos,push_type_id):
        #print('execute push action')
        push_type_id = int(push_type_id)
        x = current_pos[0]
        y = current_pos[1]
        z = current_pos[2]
        o_z = current_pos[3]
        trans_m = 0.08
        id2angle = [0,20,40,60,80,10,30,50,70,90,20,40,60,80,10,30,50,70,0,20,40,60,80,10,30,50,70,90,20,40,60,80,10,30,50,70]
        if (push_type_id%18) >= 0 and (push_type_id%18) <= 4:
            # 1
            new_x = x + trans_m * np.cos(id2angle[push_type_id] * np.pi / 180)
            new_y = y - trans_m * np.sin(id2angle[push_type_id] * np.pi / 180)
        elif (push_type_id%18) >= 5 and (push_type_id%18) <= 9:
            # 1
            new_x = x - trans_m * np.sin(id2angle[push_type_id] * np.pi / 180)
            new_y = y - trans_m * np.cos(id2angle[push_type_id] * np.pi / 180)
        elif (push_type_id%18) >= 10 and (push_type_id%18) <= 13:
            # 1
            new_x = x - trans_m * np.cos(id2angle[push_type_id] * np.pi / 180)
            new_y = y + trans_m * np.sin(id2angle[push_type_id] * np.pi / 180)
        elif (push_type_id%18) >= 14 and (push_type_id%18) <= 17:
            # 1
            new_x = x + trans_m * np.sin(id2angle[push_type_id] * np.pi / 180)
            new_y = y + trans_m * np.cos(id2angle[push_type_id] * np.pi / 180)

        #### plan push action  ####
        #workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        workspace_limits = np.asarray([[-0.7, -0.25], [-0.2, 0.2], [0.003, 0.4]])
        position = (new_x,new_y,z)

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (o_z % np.pi) - np.pi / 2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2], workspace_limits[2][0] + 0.02)  # -0.04
        position[1] = max(position[1], workspace_limits[1][0])  # -0.04
        position[1] = min(position[1], workspace_limits[1][1])
        position[0] = max(position[0], workspace_limits[0][0])  # -0.04
        position[0] = min(position[0], workspace_limits[0][1])

        # Move gripper to location above grasp target
        # grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2])

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.01 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)

            # time.sleep(0.05)#0.05

        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)



    def pick_and_hold(self,action_2):
        workspace_limits = self.workspace_limits#np.asarray([[-0.714, -0.266], [-0.224, 0.224], [-0.0001, 0.37]])
        position = (action_2[0], action_2[1], action_2[2])

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (action_2[3] % np.pi) - np.pi / 2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2], workspace_limits[2][0] + 0.02)  # -0.04

        # Move gripper to location above grasp target
        # grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2]+0.22)

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.03 * move_direction / move_magnitude  # 0.05  0.01

        # if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
        #     num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        # else:
        #     num_move_steps = 0

        ##
        if move_step[0] != 0:
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        elif move_step[1] != 0:
            num_move_steps = int(np.floor(move_direction[1] / move_step[1]))
        elif move_step[2] != 0:
            num_move_steps = int(np.floor(move_direction[2] / move_step[2]))
        ##

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)

            # time.sleep(0.05)#0.05

        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)


    def go_to_position(self,action):#
        self.close_gripper()
        # workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        position = (action[0],action[1],action[2])

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (0 % np.pi) - np.pi / 2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        #position[2] = max(position[2], workspace_limits[2][0] + 0.02)  # -0.04

        # Move gripper to location above grasp target
        # grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2])

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)

            # time.sleep(0.05)#0.05

        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

    def record_force(self):
        ##### Definition of parameters
        sensorName = 'UR5_connection'

        ##### Obtain the handle
        errorCode, returnHandle = vrep.simxGetObjectHandle(self.sim_client, sensorName, vrep.simx_opmode_blocking)#sim_ret, self.UR5_joint1_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint1', vrep.simx_opmode_blocking)
        forceSensorHandle = returnHandle

        #print('Handles available!')

        ##### Get the force sensor's force and torque
        if vrep.simxGetConnectionId(self.sim_client) != -1:
            # simx_opmode_streaming initialization, no values are read at this time
            errorCode, state, forceVector, torqueVector = vrep.simxReadForceSensor(self.sim_client, forceSensorHandle,vrep.simx_opmode_streaming)
            # ff=open('force_data.txt','a')
            # ff.write(str(forceVector)+'\n')
            # ff.close()
            if forceVector[2] >= 300 or forceVector[2] <= -300:
                time.sleep(4)
                errorCode, state, forceVector, torqueVector = vrep.simxReadForceSensor(self.sim_client,forceSensorHandle,vrep.simx_opmode_streaming)
                if forceVector[2] >= 300 or forceVector[2] <= -300:
                    ff = open('run_ep_logs.txt', 'a')
                    ff.write('********************* Error Happend, restart *******************\n')
                    ff.close()
                    self.restart_sim()

            # # Can't read twice at the same time, otherwise you can't read the value.
            # time.sleep(1)
            # # simx_opmode_buffer to obtain forceVector and torqueVector
            # errorCode, state, forceVector, torqueVector = vrep.simxReadForceSensor(self.sim_client, forceSensorHandle, vrep.simx_opmode_buffer)
            # ff=open('force_data.txt','a')
            # ff.write(str(forceVector)+'\n')
            # ff.close()
            #
            # # # Output the force of XYZ
            # # print(forceVector)
            # # # Output the torque of XYZ
            # # print(torqueVector)

    def record_force_when_close_gripper(self):
        ##### Definition of parameters
        sensorName_left = 'RG2_leftForceSensor'
        sensorName_right = 'RG2_rightForceSensor'

        ##### Obtain the handle
        errorCode, returnHandle_left = vrep.simxGetObjectHandle(self.sim_client, sensorName_left,vrep.simx_opmode_blocking)  # sim_ret, self.UR5_joint1_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint1', vrep.simx_opmode_blocking)
        errorCode, returnHandle_right = vrep.simxGetObjectHandle(self.sim_client, sensorName_right,vrep.simx_opmode_blocking)

        ##### Get the force sensor's force and torque
        if vrep.simxGetConnectionId(self.sim_client) != -1:
            # simx_opmode_streaming initialization, no values are read at this time
            errorCode, state, forceVector_l, torqueVector_l = vrep.simxReadForceSensor(self.sim_client, returnHandle_left, vrep.simx_opmode_streaming)
            errorCode, state, forceVector_r, torqueVector_r = vrep.simxReadForceSensor(self.sim_client,returnHandle_right, vrep.simx_opmode_streaming)
            ff_ = open('gripper_force.txt', 'a')
            ff_.write('*****************************************\n')
            ff_.write('left:'+str(forceVector_l)+'\n')
            ff_.write('right:' + str(forceVector_r) + '\n')
            ff_.close()

    def return_force_when_close_gripper(self):
        ##### Definition of parameters
        sensorName_left = 'RG2_leftForceSensor'
        sensorName_right = 'RG2_rightForceSensor'

        ##### Obtain the handle
        errorCode, returnHandle_left = vrep.simxGetObjectHandle(self.sim_client, sensorName_left,vrep.simx_opmode_blocking)  # sim_ret, self.UR5_joint1_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint1', vrep.simx_opmode_blocking)
        errorCode, returnHandle_right = vrep.simxGetObjectHandle(self.sim_client, sensorName_right,vrep.simx_opmode_blocking)

        ##### Get the force sensor's force and torque
        if vrep.simxGetConnectionId(self.sim_client) != -1:
            # simx_opmode_streaming initialization, no values are read at this time
            errorCode, state, forceVector_l, torqueVector_l = vrep.simxReadForceSensor(self.sim_client, returnHandle_left, vrep.simx_opmode_streaming)
            errorCode, state, forceVector_r, torqueVector_r = vrep.simxReadForceSensor(self.sim_client,returnHandle_right, vrep.simx_opmode_streaming)
            return torqueVector_l[2],torqueVector_r[2]


    def grasp(self, position, heightmap_rotation_angle, workspace_limits,grasp_flag,scene_flag):
        if scene_flag == 1:
            mm = 0.045
        if scene_flag == 2:
            mm = 0.055
        if scene_flag == 3:
            mm = 0.065

        #grasp_flag, pick is 1, place is 2
        if grasp_flag == 1:
            print('Executing: pick at (%f, %f, %f)' % (position[0], position[1], position[2]))

            #if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02) # -0.04

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.01*move_direction/move_magnitude #0.05
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)

                #time.sleep(0.05)#0.05


            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            #gripper rotation reset
            #vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, 0, np.pi/2), vrep.simx_opmode_blocking)

            # Move the grasped object elsewhere
            #if grasp_success:

                # object_positions = np.asarray(self.get_obj_positions())
                # object_positions = object_positions[:,2]
                # grasped_object_ind = np.argmax(object_positions)
                # grasped_object_handle = self.object_handles[grasped_object_ind]
                # vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)
        elif grasp_flag == 2:
            print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

            # if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] + mm, workspace_limits[2][0] + 0.02)# +0.06

            # Move gripper to location above grasp target
            grasp_location_margin = 0.15 #0.15
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)
            move_direction = np.asarray(
                [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
                 tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.01 * move_direction / move_magnitude #0.05
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                         vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
                np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                              vrep.simx_opmode_blocking)

                #time.sleep(0.05)

            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                       (tool_position[0], tool_position[1], tool_position[2]),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                          (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            #self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            #gripper_full_closed = self.close_gripper()

            # Check if grasp is successful
            self.open_gripper()
            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)

            grasp_success = True

            # Move the grasped object elsewhere
            #if grasp_success:

        return grasp_success
        # else:
        #
        #     # Compute tool orientation from heightmap rotation angle
        #     grasp_orientation = [1.0,0.0]
        #     if heightmap_rotation_angle > np.pi:
        #         heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
        #     tool_rotation_angle = heightmap_rotation_angle/2
        #     tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        #     tool_orientation_angle = np.linalg.norm(tool_orientation)
        #     tool_orientation_axis = tool_orientation/tool_orientation_angle
        #     tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
        #
        #     # Compute tilted tool orientation during dropping into bin
        #     tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
        #     tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        #     tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        #     tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])
        #
        #     # Attempt grasp
        #     position = np.asarray(position).copy()
        #     position[2] = max(position[2] - 0.05, workspace_limits[2][0])
        #     self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        #     tcp_command = "def process():\n"
        #     tcp_command += " set_digital_out(8,False)\n"
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
        #     tcp_command += " set_digital_out(8,True)\n"
        #     tcp_command += "end\n"
        #     self.tcp_socket.send(str.encode(tcp_command))
        #     self.tcp_socket.close()
        #
        #     # Block until robot reaches target tool position and gripper fingers have stopped moving
        #     state_data = self.get_state()
        #     tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #     timeout_t0 = time.time()
        #     while True:
        #         state_data = self.get_state()
        #         new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #         actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #         timeout_t1 = time.time()
        #         if (tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)])) or (timeout_t1 - timeout_t0) > 5:
        #             break
        #         tool_analog_input2 = new_tool_analog_input2
        #
        #     # Check if gripper is open (grasp might be successful)
        #     gripper_open = tool_analog_input2 > 0.26
        #
        #     # # Check if grasp is successful
        #     # grasp_success =  tool_analog_input2 > 0.26
        #
        #     home_position = [0.49,0.11,0.03]
        #     bin_position = [0.5,-0.45,0.1]
        #
        #     # If gripper is open, drop object in bin and check if grasp is successful
        #     grasp_success = False
        #     if gripper_open:
        #
        #         # Pre-compute blend radius
        #         blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
        #
        #         # Attempt placing
        #         self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        #         tcp_command = "def process():\n"
        #         tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (position[0],position[1],bin_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel,blend_radius)
        #         tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0],bin_position[1],bin_position[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel,blend_radius)
        #         tcp_command += " set_digital_out(8,False)\n"
        #         tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
        #         tcp_command += "end\n"
        #         self.tcp_socket.send(str.encode(tcp_command))
        #         self.tcp_socket.close()
        #         # print(tcp_command) # Debug
        #
        #         # Measure gripper width until robot reaches near bin location
        #         state_data = self.get_state()
        #         measurements = []
        #         while True:
        #             state_data = self.get_state()
        #             tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #             actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #             measurements.append(tool_analog_input2)
        #             if abs(actual_tool_pose[1] - bin_position[1]) < 0.2 or all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #                 break
        #
        #         # If gripper width did not change before reaching bin location, then object is in grip and grasp is successful
        #         if len(measurements) >= 2:
        #             if abs(measurements[0] - measurements[1]) < 0.1:
        #                 grasp_success = True
        #
        #     else:
        #         self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        #         tcp_command = "def process():\n"
        #         tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
        #         tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.5,self.joint_vel*0.5)
        #         tcp_command += "end\n"
        #         self.tcp_socket.send(str.encode(tcp_command))
        #         self.tcp_socket.close()
        #
        #     # Block until robot reaches home location
        #     state_data = self.get_state()
        #     tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #     while True:
        #         state_data = self.get_state()
        #         new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #         actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #         if (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #             break
        #         tool_analog_input2 = new_tool_analog_input2



    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        #if self.is_sim:

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Adjust pushing point to be on tip of finger
        position[2] = position[2] + 0.026

        # Compute pushing direction
        push_orientation = [1.0,0.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # Move gripper to location above pushing point
        pushing_point_margin = 0.1
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is closed
        self.close_gripper()

        # Approach pushing point
        self.move_to(position, None)

        # Compute target location (push to the right)
        push_length = 0.1
        target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
        push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

        # Move in pushing direction towards target location
        self.move_to([target_x, target_y, position[2]], None)

        # Move gripper to location above grasp target
        self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

        push_success = True

        # else:
        #
        #     # Compute tool orientation from heightmap rotation angle
        #     push_orientation = [1.0,0.0]
        #     tool_rotation_angle = heightmap_rotation_angle/2
        #     tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        #     tool_orientation_angle = np.linalg.norm(tool_orientation)
        #     tool_orientation_axis = tool_orientation/tool_orientation_angle
        #     tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
        #
        #     # Compute push direction and endpoint (push to right of rotated heightmap)
        #     push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
        #     target_x = min(max(position[0] + push_direction[0]*0.1, workspace_limits[0][0]), workspace_limits[0][1])
        #     target_y = min(max(position[1] + push_direction[1]*0.1, workspace_limits[1][0]), workspace_limits[1][1])
        #     push_endpoint = np.asarray([target_x, target_y, position[2]])
        #     push_direction.shape = (3,1)
        #
        #     # Compute tilted tool orientation during push
        #     tilt_axis = np.dot(utils.euler2rotm(np.asarray([0,0,np.pi/2]))[:3,:3], push_direction)
        #     tilt_rotm = utils.angle2rotm(-np.pi/8, tilt_axis, point=None)[:3,:3]
        #     tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        #     tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        #     tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])
        #
        #     # Push only within workspace limits
        #     position = np.asarray(position).copy()
        #     position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
        #     position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
        #     position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005) # Add buffer to surface
        #
        #     home_position = [0.49,0.11,0.03]
        #
        #     # Attempt push
        #     self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        #     tcp_command = "def process():\n"
        #     tcp_command += " set_digital_out(8,True)\n"
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (push_endpoint[0],push_endpoint[1],push_endpoint[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        #     tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        #     tcp_command += "end\n"
        #     self.tcp_socket.send(str.encode(tcp_command))
        #     self.tcp_socket.close()
        #
        #     # Block until robot reaches target tool position and gripper fingers have stopped moving
        #     state_data = self.get_state()
        #     while True:
        #         state_data = self.get_state()
        #         actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #         if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #             break
        #     push_success = True
        #     time.sleep(0.5)

        return push_success


    def restart_real(self):

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0,0.0]
        tool_rotation_angle = -np.pi/4
        tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation/tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5,-0.35,-0.12]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5,0.08,-0.12]
        home_position = [0.49,0.11,0.03]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2]+0.3,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.02,self.joint_vel*0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2]+0.3,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]+0.05,box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

    def cal_margin_lefttouch_to_righttouch(self):
        sensorName_left = 'RG2_leftTouch'
        sensorName_right = 'RG2_rightTouch'
        sim_ret, RG2_gripper_handle_left = vrep.simxGetObjectHandle(self.sim_client, sensorName_left,vrep.simx_opmode_blocking)
        sim_ret, RG2_gripper_handle_right = vrep.simxGetObjectHandle(self.sim_client, sensorName_right,vrep.simx_opmode_blocking)

        sim_ret, gripper_joint_position_leftTouch = vrep.simxGetObjectPosition(self.sim_client, RG2_gripper_handle_left, -1, vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position_rightTouch = vrep.simxGetObjectPosition(self.sim_client, RG2_gripper_handle_right, -1, vrep.simx_opmode_blocking)
        return np.linalg.norm(np.array(gripper_joint_position_rightTouch)-np.array(gripper_joint_position_leftTouch))


    def move_grasped_object_to_conner(self):
        #x:[-0.73,-0.27],y:[-0.23,0.23] ,block:0.048*0.048
        ####### 桌面的九个网格位置, 以中心点位置，0~0.04的reset范围
        # x (-0.1,0), y(0.2, 0.3)
        x = random.randint(-100,0) / 1000.0
        y = random.randint(200,300)/1000.0
        rp1 = (x,y,0.1)
        ro1 = (0,0,np.pi/2.0)

        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            if object_position[2] > 0.085:
                '''随机位置'''
                vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, rp1, vrep.simx_opmode_blocking)
                '''随机角度'''
                vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, ro1, vrep.simx_opmode_blocking)
                break

    def random_position(self):
        #x:[-0.73,-0.27],y:[-0.23,0.23] ,block:0.048*0.048
        ####### 桌面的九个网格位置, 以中心点位置，0~0.04的reset范围
        rp1 = (-0.36,-0.15,0.0250)
        rp2 = (-0.5, -0.15,0.0250)
        rp3 = (-0.64, -0.15,0.0250)
        rp4 = (-0.36,0,0.0250)
        rp5 = (-0.5, 0,0.0250)
        rp6 = (-0.64, 0,0.0250)
        rp7= (-0.36, 0.15,0.0250)
        rp8 = (-0.5, 0.15,0.0250)
        rp9 = (-0.64, 0.15,0.0250)
        random_opos = [rp1,rp2,rp3,rp4,rp5,rp6,rp7,rp8,rp9]
        randId = list(range(9))
        randId = random.sample(randId,9)
        random_pos = []
        for index1 in range(9):
            rx = random.randint(-40, 40)
            rx = rx/1000.0
            ry = random.randint(-40, 40)
            ry = ry / 1000.0
            random_pos.append((random_opos[randId[index1]][0]+rx,random_opos[randId[index1]][1]+ry,0.025))
            #print ("ten random position:", (random_opos[randId[index1]][0]+rx,random_opos[randId[index1]][1]+ry,0.025))

        ####### 随机生成七个位置
        # p1 = (-0.655, 0.155, 0.0250)
        # p2 = (-0.65 ,0, 0.0250)
        # p3 = (-0.65, -0.15,0.0250)
        # p4 = (-0.55,-0.15, 0.0250)
        # p5 = (-0.5, 0.145, 0.0250)
        # p6 = (-0.56,0, 0.0250)
        # p7 = (-0.43,0, 0.0250)
        # p8 = (-0.43,-0.13, 0.0250)
        # random_pos = [p1,p2,p3,p4,p5,p6,p7,p8,p8,p8]

        ####### 随机生成十个旋转角度
        random_angles = []
        angles = list(range(90))
        random_an = random.sample(angles,10)
        for an_num in random_an:
            random_angles.append((0,0,np.pi/2*(an_num/90.0)))
            #print ("ten random angles:",np.pi/2*(an_num/90.0))

        index = 0

        for object_handle in self.object_handles:
            '''随机位置'''
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, random_pos[index], vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape2_handle, -1, p2, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape3_handle, -1, p3, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape4_handle, -1, p4, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape5_handle, -1, p5, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape6_handle, -1, p6, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape7_handle, -1, p7, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, shape8_handle, -1, p8, vrep.simx_opmode_blocking)

            '''随机角度'''
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, random_angles[index],vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client, shape4_handle, -1, (np.pi / 2, - np.pi / 2, np.pi / 2),vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client, shape7_handle, -1, (np.pi / 2, 0.1, np.pi / 2),vrep.simx_opmode_blocking)

            index+=1
