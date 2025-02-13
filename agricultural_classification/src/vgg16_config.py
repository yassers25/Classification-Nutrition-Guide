class VGG16Config:
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Model parameters
    IMG_SIZE = 224  # VGG16 expected input size
    
    # Paths
    MODEL_SAVE_DIR = 'models'
    
    @staticmethod
    def get_classes():
        return ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
                'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
                'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika',
                'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
                'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
                'turnip', 'watermelon']