from PIL import Image
from torch.nn import Sequential, Softmax
from fastai.vision import load_learner


def prep_image(image, transforms):
    """Prepare image for inference by resizing, converting to tensor and normalizing."""
    img = Image.open(image)
    tensor = transforms(img)
    return tensor[None, :, :, :]
    
def get_model_and_classes(pkl_location, pkl_name, add_softmax=True):
    """Get the trained pytorch model and classes.""" 
    learner = load_learner(pkl_location, pkl_name)
    classes = learner.data.classes
    model = learner.model
    model.training = False
    model.cpu()
    model = Sequential(model, Softmax(1))
    return model, classes
