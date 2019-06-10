from io import BytesIO
from PIL import Image
from torchvision import transforms
from torch.nn import Sequential, Softmax
from fastai.vision import load_learner

import os

LEARNER_PKL_LOC = "models/"
LEARNER_PKL_NAME = "TypeClass_ResNet50.pkl"

# Need to install:
#  Pillow
#  fastai
#  torch (cpu only)


# For data preprocessing I need to normalize the images to imagenet stats and
# change the resolution to 224 x 224

# For the model it may be possible to create the fastai model, load it's weights
# and then take the pytorch model from that. (Easier, but leaves fastai dependency)

# Also, for lighter docker image use the CPU ONLY version of pytorch

class TypeClassifier:
    """Clothing image classifier that uses pytorch CNN model trained with Fast.ai library."""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.model, self.classes = self._get_model_and_classes()

    def classify(self, file):
        """Classify given image stream into one of clothing type categories. Return a dict of probabilites"""
        preped_img = self._prep_image(file)
        prediction_tensor = self.model(preped_img)
        predictions = prediction_tensor.data.cpu().numpy()[0].tolist()
        return {"predictons": predictions, "classes": self.classes}
      
    def _prep_image(self, image):
        """Prepare image for inference by resizing, converting to tensor and normalizing"""
        img = Image.open(image)
        tensor = self.transform(img)
        return tensor[None, :, :, :]
        
    def _get_model_and_classes(self):
        learner = load_learner(LEARNER_PKL_LOC, LEARNER_PKL_NAME)
        classes = learner.data.classes
        model = learner.model
        model.training = False
        model.cpu()
        model = Sequential(model, Softmax(1))
        return model, classes