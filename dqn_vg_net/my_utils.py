import numpy as np
import random
from configs import Configs
from PIL import Image
import cv2

_configs = Configs()


def depth_npy2mask():
    pass

def actionID_to_depthImgXY(action_id):
    dim_for_one_direction = 224 * 224
    angle_id = int(action_id / dim_for_one_direction)
    action_oz = angleId_to_angle(angle_id)
    # trans to depth imageXY
    new_id = action_id % dim_for_one_direction
    h = int(new_id / 224) + 1
    w = new_id % 224 + 1
    # (col,row)
    depth_x, depth_y = w, h
    return (depth_x,depth_y)#(col,row)
def depthImgXY_to_actionID_withRandAngle(pos):
    dim_for_one_direction = 224 * 224
    h = pos[1]
    w = pos[0]
    id_temp = (w-1) + (h-1)*224
    rand_angle = np.random.randint(0,10)
    id = id_temp + rand_angle * dim_for_one_direction
    return int(id)

def predict_actionID_to_execution_action(action_id,current_depth_path):
    # actionID [0~224*224*10-1]
    # the id increase from top to bottom and from left to right, and return the position(x,y) in format (w,h), [1~224]
    dim_for_one_direction = 224*224
    angle_id = int(action_id/dim_for_one_direction)
    action_oz = angleId_to_angle(angle_id)
    # trans to depth imageXY
    new_id = action_id%dim_for_one_direction
    h = int(new_id/224) + 1
    w = new_id%224 + 1
    #(col,row)
    depth_x,depth_y = w,h
    action_x, action_y = imageXY_to_robotXY((depth_x, depth_y))
    action_z = cal_corresponding_z_fromXY((depth_x, depth_y), current_depth_path)

    # return_x = work_space_x[1] - (p[0] - 1) * (abs(work_space_x[1] - work_space_x[0]) / 226.0)
    # return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 226.0)

    # return (return_x, return_y)  # imageXY format: (width,height)
    action = (action_x, action_y, action_z, action_oz)
    return action

def cal_corresponding_z_fromXY(XY,depth_path):#(col,row)
    # print('XY:',XY)
    data=np.load(depth_path)
    depth_value = data[int(XY[0])-1][int(XY[1])-1]

    Z = (0.90788731274057 - depth_value) / 1.07428894499597
    t = Z/2.0 - 0.0065
    if t >= 0.01:
        re = t
    else:
        re = 0.01
    return re

def select_randP_from_mask(npy_path):
    # no-object value:0.908,0.907
    # object value:0.86
    xy_contrain_object = []  # record the pixel of object
    data = np.load(npy_path)
    h, w = data.shape
    for i in range(w):
        for j in range(h):
            if data[i][j] < 0.896:
                xy_contrain_object.append((j, i))
    if len(xy_contrain_object) != 0:
        rand_id = random.randint(0, len(xy_contrain_object) - 1)
        return xy_contrain_object[rand_id]# return imageXY format: (width,height)
    else:
        print('there is not valid point detected !')
        # return (random.randint(10,217),random.randint(10,217))
        return -1

def select_randpID_from_mask(npy_path):
    p = select_randP_from_mask(npy_path)
    if p == -1:
        return np.random.randint(0,_configs.DIM_ACTIONS)
    else:
        id = depthImgXY_to_actionID_withRandAngle(p)
        return id

def imageXY_to_robotXY(p):
    # work_space = _configs.WORKSPACE_LIMITS
    # work_space_x = work_space[0]
    # work_space_y = work_space[1]
    #
    # return_x = work_space_x[0] + (p[0] - 1) * -(abs(work_space_x[1] - work_space_x[0]) / 226.0)
    # return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 226.0)
    #
    # return (return_x,return_y)
    # WORKSPACE_LIMITS = np.asarray([[-0.705, -0.195], [-0.255, 0.255], [0.01, 0.3]])
    work_space = _configs.WORKSPACE_LIMITS
    work_space_x = work_space[0]
    work_space_y = work_space[1]

    return_x = work_space_x[1] - (p[0] - 1) * (abs(work_space_x[1] - work_space_x[0]) / 223.0)
    return_y = work_space_y[0] + (p[1] - 1) * ((work_space_y[1] - work_space_y[0]) / 223.0)

    return (return_x, return_y) # imageXY format: (width,height)

def robotXY_to_imageXY(p):
    # work_space = _configs.WORKSPACE_LIMITS
    # work_space_x = work_space[0]
    # work_space_y = work_space[1]
    #
    #
    # return_y = (226 * (p[0]-work_space_x[0]) + (work_space_x[1] - work_space_x[0])) / (work_space_x[1] - work_space_x[0])#action_space_x[0] + (p[0] - 1) * ((action_space_x[1] - action_space_x[0]) / 226.0)
    # return_x = (226 * (p[1]-work_space_y[0]) + (work_space_y[1] - work_space_y[0])) / (work_space_y[1] - work_space_y[0])
    #
    # return (return_x,return_y)
    work_space = _configs.WORKSPACE_LIMITS
    work_space_x = work_space[0]
    work_space_y = work_space[1]

    return_w = ((abs(p[0]) - abs(work_space_x[1])) / abs(work_space_x[1] - work_space_x[0])) * 224
    return_h = (p[1] - work_space_y[0]) / abs(work_space_y[1] - work_space_y[0]) * 224
    if return_w == 0:
        return_w = 1
    if return_h == 0:
        return_h = 1

    return (return_w, return_h)  # imageXY format: (width,height)

def output_action_2_execution_action(x, y):
    real_x = _configs.ACTION_SPACE_GRASP[0][1] - 0.5 * (1 - x) * (_configs.ACTION_SPACE_GRASP[0][1] - _configs.ACTION_SPACE_GRASP[0][0])
    real_y = _configs.ACTION_SPACE_GRASP[1][1] - 0.5 * (1 - y) * (_configs.ACTION_SPACE_GRASP[1][1] - _configs.ACTION_SPACE_GRASP[1][0])

    return real_x, real_y


def execution_action_2_output_action(x, y):
    out_x = 1 - 2.0 * (_configs.ACTION_SPACE_GRASP[0][1] - x) / (_configs.ACTION_SPACE_GRASP[0][1] - _configs.ACTION_SPACE_GRASP[0][0])
    out_y = 1 - 2.0 * (_configs.ACTION_SPACE_GRASP[1][1] - y) / (_configs.ACTION_SPACE_GRASP[1][1] - _configs.ACTION_SPACE_GRASP[1][0])

    return out_x, out_y


def angleId_to_angle(id):# the range (0~180), not (0~360)
    temp = 18 * id
    return temp * np.pi / 180.0

def angle_to_angleId(ang):# the range (0~180), not (0~360)
    angle_index = np.array(list(range(16))) / 16.0 * np.pi
    # print('id:',np.argmin(abs(angle_index-ang)))
    return np.argmin(abs(angle_index-ang))

#----------------- functions for multi-channel FCN DQN -------------------
def translateXY_and_channel_to_actionID(x,y,c):
    '''
    given a state, the eval_net produce the multi-channel q-value map (c,224,224), then find the max q-value, get its position(x,y) and channel c
    :param x: h, the row
    :param y: w, the col
    :param c: channel
    :return: id, rule: from left to right, from top to bottom, the id is (0~224*224*c)
    '''
    im_size = _configs.DIM_STATES[0]
    margin = im_size*im_size
    return x * im_size + y + margin*c

def translate_actionID_to_XY_and_channel(action_id):
    '''
    action_id => (x,y,c)
    :param x: h, the row
    :param y: w, the col
    :param c: channel, the id of input depth
    :return: id, rule: from left to right, from top to bottom, the id is (0~224*224*c)
    '''
    im_size = _configs.DIM_STATES[0]
    margin = im_size*im_size
    c = int(action_id/margin)
    action_id0 = action_id%margin
    x = int(action_id0/im_size)
    y = action_id0 % im_size
    return x,y,c
def translate_actionID_to_XY_and_channel_batch(batch_action_id):
    result_list = []
    for action_id in batch_action_id:
        [x,y,c] = translate_actionID_to_XY_and_channel(action_id)
        result_list.append([x,y,c])
    return result_list
def find_maxQ_in_qmap(q_map):
    '''
    :param q_map: [INPUT_IMAGES,1,224,224] #[CHANNEL,224,224]
    :return: action_id with the max q value
    '''

    a = q_map.reshape((_configs.ROTATION_BINS,_configs.DIM_STATES[0],_configs.DIM_STATES[1]))
    m, n, l = a.shape
    index = int(a.argmax())
    x = int(index / (n * l))
    index = index % (n * l)
    y = int(index / l)
    index = index % l
    z = index
    return translateXY_and_channel_to_actionID(y,z,x)#(x, y, z)
def copy_depth_to_3_channel(state_path):
    if '.npy' in str(state_path):
        origin_depth_arr = np.load(state_path)
        #return np.stack((one_dim_arr, one_dim_arr, one_dim_arr))
    else:
        origin_depth_arr = state_path
        #return np.stack((state_path, state_path, state_path))

    # mask
    mask_arr = (origin_depth_arr - 0.9) * -1000000
    mask_arr = np.clip(mask_arr, 0, 255)
    mask_arr = mask_arr / 255.0
    # heightmap
    heightmap_arr = get_heightmap_from_depth(origin_depth_arr, mask_arr)
    return np.stack((heightmap_arr, heightmap_arr, heightmap_arr))


def get_heightmap_from_depth(depth_arr,mask_arr):
    height_map_arr = mask_arr
    bottom_mean = 0.9078791
    h_inds,w_inds = np.where(mask_arr==255)
    for i in range(len(h_inds)):
        height_map_arr[h_inds[i]][w_inds[i]] = round(( bottom_mean - mask_arr[h_inds[i]][w_inds[i]] ) *1000) # mm level

    height_map_arr = (height_map_arr - _configs.MIN_HEIGHTMAP_ARR) / (_configs.MAX_HEIGHTMAP_ARR - _configs.MIN_HEIGHTMAP_ARR)
    return np.clip(height_map_arr, 0, 100)

#----------------- functions for rotate single-channel FCN DQN -------------------
def trans_HWC_to_CHW(im_rgb):
    if type(im_rgb) == type('path'):
        im_rgb = cv2.imread(im_rgb)
    img = im_rgb / 255.0  # BGR
    img_B = img[:, :, 0]
    img_G = img[:, :, 1]
    img_R = img[:, :, 2]
    img_T = np.array([img_R, img_G, img_B])
    return img_T

def rotate_bound(image, angle):#顺时针旋转20
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))

def fix_black_empty(image):
    # np.save('rotate.npy',image)
    max_pixel_value = np.max(image)
    h,w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i][j] <= 0.75:
                image[i][j] = max_pixel_value
    return image

def get_rotate_depth(rotate_id,depth_npy_path):
    data=np.load(depth_npy_path)
    bin_angle = 180 / _configs.ROTATION_BINS
    rotate_angle = bin_angle*rotate_id
    data_rotate = rotate_bound(data, rotate_angle)
    # data_fix = fix_black_empty(data_rotate)
    # data_resize = cv2.resize(data_fix, _configs.DIM_STATES, interpolation=cv2.INTER_AREA)
    #------ center crop --------
    h,w = data_rotate.shape
    h_norm, w_norm = _configs.DIM_STATES
    h_start = int(h/2) - int(h_norm/2)
    h_end = h_start + h_norm
    w_start = int(w/2) - int(w_norm/2)
    w_end = w_start + w_norm
    data_resize = data_rotate[h_start:h_end,w_start:w_end]
    #---- test code -----
    # cv2.imwrite('testimg/'+depth_npy_path.replace('state_depth/','').replace('.npy','_dep'+str(rotate_id)+'.png'),data_resize*150)

    return data_resize

def get_rotate_rgb(rotate_id,rgb_path):
    data=cv2.imread(rgb_path)
    bin_angle = 180 / _configs.ROTATION_BINS
    rotate_angle = bin_angle*rotate_id
    data_rotate = rotate_bound(data, rotate_angle)
    # data_resize = cv2.resize(data_rotate, _configs.DIM_STATES, interpolation=cv2.INTER_AREA)
    h, w, c = data_rotate.shape
    h_norm, w_norm = _configs.DIM_STATES
    h_start = int(h / 2) - int(h_norm / 2)
    h_end = h_start + h_norm
    w_start = int(w / 2) - int(w_norm / 2)
    w_end = w_start + w_norm
    data_resize = data_rotate[h_start:h_end, w_start:w_end, :]
    #---- test code -----
    # cv2.imwrite('testimg/'+rgb_path.replace('state_image/','').replace('.png','_'+str(rotate_id)+'.png'),data_resize)

    return trans_HWC_to_CHW(data_resize)