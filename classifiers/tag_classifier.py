from torchvision import transforms
from classifiers.utils import get_model_and_classes, prep_image


LEARNER_PKL_LOC = "models/"
LEARNER_PKL_NAME = "TagClass_ResNet50.pkl"


class TagClassifier:
    """Chlothing material pattern classifier that uses pytorch CNN model trained with Fast.ai library."""

    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.model, self.classes = get_model_and_classes(LEARNER_PKL_LOC, LEARNER_PKL_NAME)

    def classify(self, file):
        """Classify given image stream into one of pattern categories. Return a dict of probabilites"""
        preped_img = prep_image(file, self.transform)
        prediction_tensor = self.model(preped_img)
        predictions = prediction_tensor.data.cpu().numpy()[0].tolist()
        return {"predictons": predictions, "classes": self.classes}
