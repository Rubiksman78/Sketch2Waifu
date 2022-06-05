config = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval, 5: drawing
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console

    'INPUT_SIZE': 128,              # input image size for training 0 for original size
    'SIGMA': 2.5,                     # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'KM': 3,
}