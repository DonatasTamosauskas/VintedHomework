from io import BytesIO
from PIL import Image
from torchvision import transforms
from fastai.vision import load_learner

LEARNER_PKL_LOC = "data/"

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
      
      self.model = self._get_model()

    def classify(self, file):
        """Classify given image stream into one of clothing type categories. Return a dict of probabilites"""
        preped_img = self._prep_image(file)
        # Might need to alter dimensions to signal batch of one entity
        predictions = self.model(preped_img)
        return predictions
      
    def _prep_image(self, image):
      """Prepare image for inference by resizing, converting to tensor and normalizing"""
      img = Image.open(BytesIO(image))
      tensor = self.transform(img)
      return tensor[None, :, :, :]
    
    def _get_model(self):
      learner = load_learner(LEARNER_PKL_LOC)
      model = learner.model
      model.training = False
      model.cpu()
      return model