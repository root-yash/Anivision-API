import base64
import torch
import numpy as np
import cv2
from torchvision import transforms
import joblib
from utils import ctob, nms


class YoLo:
    def __init__(self, image):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(256,256)),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        im_bytes = base64.b64decode(image)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        self.image = transformer(img).reshape((1,3,256,256))
        self.anchor = torch.tensor([[[0.2788, 0.2163], [0.3750, 0.4760], [0.8966, 0.7837]],
                                    [[0.0721, 0.1466], [0.1490, 0.1082], [0.1418, 0.2861]],
                                    [[0.0240, 0.0312], [0.0385, 0.0721], [0.0793, 0.0553]]])

    def load_model(self):
        return torch.jit.load('Model/model_398_trace.pt')

    def load_classes(self):
        return joblib.load('Model/class_dict.pickle')

    def predict(self):
        model = self.load_model()
        model.eval()
        with torch.no_grad():
            prediction = model(self.image)
        return prediction

    def getbbox(self):
        prediction = self.predict()
        bboxes = ctob(prediction, self.anchor)
        prediction = nms(bboxes[0],self.load_classes(), 0.5, 0.5)
        return prediction

def predict_box(image):
    model = YoLo(image)
    result = model.getbbox()
    return result





