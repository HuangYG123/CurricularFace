import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        DATA_ROOT = '/data2/yugehuang/data/', # the parent root where your train/val/test data are stored
        RECORD_DIR = '/data2/yugehuang/data/refined_ms1m.txt', # the dataset record dir
        MODEL_ROOT = './train_log/model', # the root to buffer your checkpoints
        LOG_ROOT = './train_log/log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = "",
        HEAD_RESUME_ROOT = "",
        BACKBONE_NAME = 'IR_101', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = "CurricularFace", # support:  ['ArcFace', 'CurricularFace']
        LOSS_NAME = 'Softmax', # support: ['Focal', 'Softmax']
        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        LR = 0.1, # initial LR
        START_EPOCH = 0, #start epoch
        NUM_EPOCH = 24, # total epoch number
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [10, 18, 22], # ms1m epoch stages to decay learning rate
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = 0, # specify your GPU ids
        DIST_BACKEND = 'nccl',
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 5,
        TEST_GPU_ID = [0,1,2,3,4,5,6,7]
    ),
}
