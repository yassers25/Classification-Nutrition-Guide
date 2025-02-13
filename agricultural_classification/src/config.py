import os

class Config:
    # Chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'validation')
    TEST_DIR = os.path.join(BASE_DIR, 'test')
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
    
    # Paramètres du modèle
    MODEL_NAME = 'resnet50'
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Réduit car petit dataset
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Classes (générées automatiquement)
    @staticmethod
    def get_classes():
        return sorted([d for d in os.listdir(Config.TRAIN_DIR) 
                      if os.path.isdir(os.path.join(Config.TRAIN_DIR, d))])