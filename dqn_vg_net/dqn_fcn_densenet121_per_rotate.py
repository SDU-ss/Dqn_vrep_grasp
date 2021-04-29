# ------------------------------------------------------------------------------------------------------------
# Deep Q-Learning Algorithm for learning graspable affordance
# Input: color + depth image (RGBD)
# Version: pytorch code
# !! With Prioritized Experience Replay (PER)
# !! Using FCN(DenseNet121) output single-channel q-value affordance map,
#    and input 8 rotated depth images at same time, then find the max q-value in all 8 output q-value map
# !! Using pre-trained weights
# ------------------------------------------------------------------------------------------------------------

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import shutil
from vrep_env import ArmEnv
import random
from configs import Configs
import my_utils
import shutil
from tensorboardX import SummaryWriter
from prioritized_memory import Memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#set hyper parameters
Train_Configs = Configs()

env = ArmEnv()

DIM_ACTIONS = Train_Configs.DIM_ACTIONS
DIM_STATES = Train_Configs.DIM_STATES
CHANNELS = Train_Configs.ANGLE_CHANNELS

#--------------------------------------
# Build network for q-value prediction
# Input: RGB-D, dim:[3,224,224]
# Output: [1,224,224]
#--------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model_rgb = torchvision.models.densenet121(pretrained=True)
        self.get_feature_rgb = self.base_model_rgb.features  # shape:([batch_size, 1024, 7, 7]),input[batch_size,3,224,224]
        self.base_model_depth = torchvision.models.densenet121(pretrained=True)
        self.get_feature_depth = self.base_model_depth.features

        self.conv_feat = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, CHANNELS, kernel_size=1, stride=1, bias=False)
        )#out:[batch_size,CHANNELS,7,7]

        # Lateral convolutional layer
        self.lateral_layer = nn.Conv2d(3, CHANNELS, kernel_size=1, stride=1, padding=0)# 512
        # Bilinear Upsampling
        self.up = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, x_rgb, x_depth):
        out_rgb = self.get_feature_rgb(x_rgb) # out:[batch_size,1024,7,7]
        out_depth = self.get_feature_depth(x_depth)  # out:[batch_size,1024,7,7]
        concat_feat = torch.cat((out_rgb, out_depth), dim=1) # out:[batch_size,2048,7,7]
        out2 = self.conv_feat(concat_feat) # out:[batch_size,CHANNELS,7,7]
        out_up =  self.up(out2) + self.lateral_layer(x_depth)
        return out_up#dim:[batch_size,CHANNELS,224,224]

class Dqn():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.cuda()
        self.target_net.cuda()

        # create prioritized replay memory using SumTree
        self.memory = Memory(Train_Configs.MEMORY_CAPACITY)
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Train_Configs.LR,betas=(0.9, 0.99), eps=1e-08, weight_decay=2e-5)
        self.loss = nn.MSELoss(reduce=False, size_average=False)

        self.fig, self.ax = plt.subplots()
        self.discount_factor = Train_Configs.GAMMA

    def store_trans(self, state_path, action, reward, next_state_path,done):
        ## action type: id
        x, y, c = my_utils.translate_actionID_to_XY_and_channel(action)
        trans = state_path+'#'+str(action)+'#'+str(reward)+'#'+next_state_path#np.hstack((state, [action], [reward], next_state))
        #------ calculate TD errors from (s,a,r,s'), #--only from the first depth image, without considering other 9 rotated depth images
        state_d = state_path
        next_state_d = next_state_path
        if c > 0:
            state_d = my_utils.get_rotate_depth(c,state_d)
            next_state_d = my_utils.get_rotate_depth(c, next_state_d)
        state_depth = my_utils.copy_depth_to_3_channel(state_d).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        next_state_depth = my_utils.copy_depth_to_3_channel(next_state_d).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])

        if c == 0:
            state_rgb = my_utils.trans_HWC_to_CHW(cv2.imread(state_path.replace('npy','png').replace('state_depth','state_image'))).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
            next_state_rgb = my_utils.trans_HWC_to_CHW(cv2.imread(next_state_path.replace('npy','png').replace('state_depth', 'state_image'))).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        else:
            state_rgb = my_utils.get_rotate_rgb(c,state_path.replace('npy','png').replace('state_depth','state_image')).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
            next_state_rgb = my_utils.get_rotate_rgb(c,next_state_path.replace('npy','png').replace('state_depth','state_image')).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])

        # # normlize
        # state_depth = (state_depth - Train_Configs.MIN_DEPTH_ARR) / (Train_Configs.MAX_DEPTH_ARR - Train_Configs.MIN_DEPTH_ARR)
        # next_state_depth = (next_state_depth - Train_Configs.MIN_DEPTH_ARR) / (Train_Configs.MAX_DEPTH_ARR - Train_Configs.MIN_DEPTH_ARR)
        # numpy to tensor
        state_depth = torch.cuda.FloatTensor(state_depth)
        next_state_depth = torch.cuda.FloatTensor(next_state_depth)
        state_rgb = torch.cuda.FloatTensor(state_rgb)
        next_state_rgb = torch.cuda.FloatTensor(next_state_rgb)

        target_singleChannel_q_map = self.eval_net.forward(state_rgb,state_depth)#dim:[1,1,224,224],CHANNEL=1
        # x,y,c = my_utils.translate_actionID_to_XY_and_channel(action)
        old_val = target_singleChannel_q_map[0][0][x][y]
        # old_val = target[0][action]
        target_val_singleChannel_q_map = self.target_net.forward(next_state_rgb,next_state_depth)#dim:[1,1,224,224]

        if done == 1:
            target_q = reward # target[0][action] = reward
        else:
            target_q = reward + self.discount_factor * torch.max(target_val_singleChannel_q_map) # target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target_q)
        self.memory.add(float(error), trans)

    def choose_action(self, state_path,EPSILON):
        state_rgb = []
        state_depth = []
        state_rgb.append(my_utils.trans_HWC_to_CHW(cv2.imread(state_path.replace('npy','png').replace('state_depth','state_image'))))
        state_depth.append(my_utils.copy_depth_to_3_channel(state_path))#dim:[3, DIM_STATES[0], DIM_STATES[1]]#.reshape(1, 3, DIM_STATES[0], DIM_STATES[1]))
        for i in range(1,Train_Configs.ROTATION_BINS):
            state_rotate_rgb = my_utils.get_rotate_rgb(i,state_path.replace('npy','png').replace('state_depth','state_image'))
            state_rgb.append(state_rotate_rgb)
            #------------------------
            state_rotate_depth = my_utils.get_rotate_depth(i,state_path)
            state_rotate_3_depth = my_utils.copy_depth_to_3_channel(state_rotate_depth)
            state_depth.append(state_rotate_3_depth)

        state_rgb = np.array(state_rgb)
        state_depth = np.array(state_depth)
        # # normlize
        # state_depth = (state_depth - Train_Configs.MIN_DEPTH_ARR) / (Train_Configs.MAX_DEPTH_ARR - Train_Configs.MIN_DEPTH_ARR)
        # numpy to tensor
        state_rgb = torch.cuda.FloatTensor(state_rgb)  # dim:[INPUT_IMAGE,3,224,224]
        state_depth = torch.cuda.FloatTensor(state_depth) #dim:[INPUT_IMAGE,3,224,224]

        # random exploration
        prob = np.min((EPSILON,1))
        p_select = np.array([prob, 1 - prob])
        selected_ac_type = np.random.choice([0, 1], p=p_select.ravel())

        if selected_ac_type == 0:#origin predicted action
            target_multiChannel_q_map = self.eval_net.forward(state_rgb,state_depth)  # dim:[INPUT_IMAGES,1,224,224]
            action = my_utils.find_maxQ_in_qmap(target_multiChannel_q_map.cpu().detach().numpy())
            ac_ty = '0'
        else:
            if np.random.randn() <= 0.5:#sample action according to depth image
                action = my_utils.select_randpID_from_mask(state_path)
                ac_ty = '1'
            else:# random sample
                action = np.random.randint(0,DIM_ACTIONS)
                ac_ty = '2'

        return ac_ty,action # the id of action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def load_batch_data(self,batch_list):#batch_list.dim:[batch_size]
        # print(batch_list)
        batch_state_rgb = []
        batch_state_depth = []
        batch_action = []
        batch_reward = []
        batch_next_state_rgb = []
        batch_next_state_depth = []

        for item in batch_list:
            data = item.split('#')#state+'#'+str(action)+'#'+str(reward)+'#'+next_state
            action_id = int(data[1])
            batch_state_rgb.append(my_utils.get_rotate_rgb(action_id,data[0].replace('npy','png').replace('state_depth','state_image')))
            batch_state_depth.append(my_utils.copy_depth_to_3_channel(my_utils.get_rotate_depth(action_id,data[0])).reshape((3,DIM_STATES[0],DIM_STATES[1])))
            batch_action.append([int(data[1])])
            batch_reward.append([float(data[2])])
            batch_next_state_rgb.append(my_utils.get_rotate_rgb(action_id, data[3].replace('npy','png').replace('state_depth', 'state_image')))
            batch_next_state_depth.append(my_utils.copy_depth_to_3_channel(my_utils.get_rotate_depth(action_id,data[3])).reshape((3,DIM_STATES[0],DIM_STATES[1])))

        batch_state_depth = np.array(batch_state_depth)
        batch_next_state_depth = np.array(batch_next_state_depth)
        # # normlize
        # batch_state_depth = (batch_state_depth - Train_Configs.MIN_DEPTH_ARR) / (Train_Configs.MAX_DEPTH_ARR - Train_Configs.MIN_DEPTH_ARR)
        # batch_next_state_depth = (batch_next_state_depth - Train_Configs.MIN_DEPTH_ARR) / (Train_Configs.MAX_DEPTH_ARR - Train_Configs.MIN_DEPTH_ARR)

        return torch.cuda.FloatTensor(batch_state_rgb),torch.cuda.FloatTensor(batch_state_depth),torch.cuda.LongTensor(batch_action),torch.cuda.FloatTensor(batch_reward),torch.cuda.FloatTensor(batch_next_state_rgb),torch.cuda.FloatTensor(batch_next_state_depth)

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Train_Configs.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1

        mini_batch, idxs, is_weights = self.memory.sample(Train_Configs.BATCH_SIZE)#
        batch_state_rgb,batch_state_depth,batch_action,batch_reward,batch_next_state_rgb,batch_next_state_depth = self.load_batch_data(mini_batch)#dim:[1]

        eval_singleChannel_q_map = self.eval_net(batch_state_rgb,batch_state_depth)  # dim:[BATCH_SIZE,1,224,224]
        x_y_c_list = my_utils.translate_actionID_to_XY_and_channel_batch(batch_action)
        # old_val = target_multiChannel_q_map[0][c][x][y]
        batch_q = []
        # for xyc in x_y_c_list:
        for i in range(len(x_y_c_list)):
            xyc = x_y_c_list[i]
            batch_q.append([eval_singleChannel_q_map[i][0][xyc[0]][xyc[1]]])
        q_eval = torch.cuda.FloatTensor(batch_q)#self.eval_net(batch_state).gather(1, batch_action)#action: a value in range [0,DIM_ACTIONS-1]
        q_eval = Variable(q_eval.cuda(), requires_grad=True)
        target_singleChannel_q_map = self.target_net(batch_next_state_rgb,batch_next_state_depth).cpu().detach().numpy()#q_next,dim:[BATCH_SIZE,1,224,224]
        batch_q_next = []
        for b_item in target_singleChannel_q_map:#dim:[1,224,224]
            batch_q_next.append([np.max(b_item)])
        q_next = torch.cuda.FloatTensor(batch_q_next)
        # q_next = Variable(q_next.cuda(), requires_grad=True)

        q_target = batch_reward + Train_Configs.GAMMA*q_next
        q_target = Variable(q_target.cuda(), requires_grad=True)
        # self.average_q = q_eval.mean()
        weight_tensor = torch.cuda.FloatTensor(is_weights)#
        weight_tensor = weight_tensor.reshape((Train_Configs.BATCH_SIZE,1))
        weight_tensor = Variable(weight_tensor.cuda(), requires_grad=False)

        loss = (weight_tensor * self.loss(q_eval, q_target)).mean()##(torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss),float(q_eval.mean())

def main():
    torch.backends.cudnn.enabled = False
    net = Dqn()
    # net.cuda()
    EPSILON = Train_Configs.EPSILON
    print("Start, the DQN is collecting experience...")
    step_counter_list = []
    log_writer = SummaryWriter('logs/')  # ('logs/')
    log_txt_writer = open('logs/train_log.txt','a')
    step_counter = 0
    time_start = time.time()
    for episode in range(Train_Configs.EPISODES):
        state,obj_num = env.reset()
        sum_reward = 0
        step_count_every_ep = 0
        success_grasp_obj_num = 0
        while True:
            step_counter += 1
            step_count_every_ep += 1
            ac_type, action = net.choose_action(state,EPSILON)
            EPSILON = Train_Configs.EPSILON + step_counter*1e-6
            next_state, reward, done = env.step(ac_type,action,state,step_counter)
            if done == -1:
                print('Env error occured, restart simulation...')
                break

            net.store_trans(state, action, reward, next_state,done)
            sum_reward += reward
             
            if net.memory.tree.n_entries >= 1000:#1000,1100,1100
                l, mean_q = net.learn()
                if step_counter >= 1100:
                    if step_counter == 1100:
                        time_start = time.time()
                    log_writer.add_scalar('loss', float(l), global_step=step_counter)
                    log_writer.add_scalar('mean_q', float(mean_q), global_step=step_counter)
                    log_txt_writer.write('used time:'+str((time.time()-time_start)/60)+',step:'+str(step_counter)+',loss:'+str(float(l))+',mean_q:'+str(float(mean_q)))
                    log_txt_writer.write('\n')
                    #time format,hour
                # print('train network, episode:',episode,', step:',step_counter,', loss:',l)

            if  net.learn_counter % 1000 == 0 and net.learn_counter > 0:
                torch.save(net.eval_net.state_dict(), 'models/step_' + str(net.learn_counter) + '_params.pkl')
                print('#####################   save model   #####################')

            if done == 1:
                success_grasp_obj_num += 1
            if success_grasp_obj_num == obj_num or step_count_every_ep == 100:
                print("finish episode {}, the sum reward is {}".format(episode, round(sum_reward, 4)))
                break

            state = next_state

        # if (episode+1) % 200 == 0 and step_counter >= 1200:
        #     torch.save(net.eval_net.state_dict(), 'models/ep_' + str(episode+1) + '_params.pkl')
        #     print('#####################   save model   #####################')

    torch.save(net.eval_net.state_dict(), 'models/final_params.pkl')
    log_txt_writer.close()

if __name__ == '__main__':
    main()
