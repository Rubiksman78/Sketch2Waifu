DEFAULT_CONFIG = {
    'INPUT_SIZE': 128,              # input image size for training 0 for original size
    'SIGMA': 2.5,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'KM': 3,
    'NUM_SAMPLES':20, 
    'NUM_CHANNELS':4,
    'LR':1e-3,
    'N_EPOCHS':5,
    'BATCH_SIZE':4,
}