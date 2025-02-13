import os
from config import Config

class EfficientNetConfig(Config):
    # Inherit from base Config but override specific parameters
    MODEL_NAME = 'efficientnet_v2_s'
    IMG_SIZE = 384  # EfficientNet uses different image size
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001