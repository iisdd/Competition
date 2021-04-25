'''
    配置环境,训练结果神经网络参数保存在model_saved文件中
'''
import os
import torch


'''train'''
batch_size = 32
max_explore_iterations = 5000
max_memory_size = 100000
max_train_iterations = 5000000
save_interval = 10000       # 10000步存一次
save_dir = 'model_saved'
frame_size = None
num_continuous_frames = 1
logfile = 'train.log'
use_cuda = torch.cuda.is_available()
eps_start = 1.0             # 最开始的随机概率
eps_end = 0.1               # 最后的随机概率
eps_num_steps = 10000

'''test'''
#weightspath = os.path.join(save_dir, str(max_train_iterations)+'.pkl') # 存放神经网络参数的路径 , 根据最后训练成果改
weightspath = os.path.join(save_dir, str(150000)+'.pkl')

'''game'''
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
SKYBLUE = (0, 191, 255)
layout_filepath = 'layouts/mediumClassic.lay' # 为了减小训练量,选择游戏地图为mediumClassic中等大小,2个鬼.4个鬼的为trickyClassic
ghost_image_paths = [(each.split('.')[0], os.path.join(os.getcwd(), each)) for each in ['gameAPI/images/Blinky.png', 'gameAPI/images/Inky.png', 'gameAPI/images/Pinky.png', 'gameAPI/images/Clyde.png']]
scaredghost_image_path = os.path.join(os.getcwd(), 'gameAPI/images/scared.png')
pacman_image_path = ('pacman', os.path.join(os.getcwd(), 'gameAPI/images/pacman.png'))
font_path = os.path.join(os.getcwd(), 'gameAPI/font/ALGER.TTF')
grid_size = 32
operator = 'ai'                 # 'ai'表示电脑学习,可选'person'手动操作
ghost_action_method = 'random'  # 为了加速收敛,鬼的行为定为随机