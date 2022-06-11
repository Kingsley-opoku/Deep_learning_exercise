import torch 
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from model import CovNetMNIST

model=CovNetMNIST()
model = torch.load('mymodel1.pth')

class NumberPredictor:

    def __init__(self, model, img_path: str, n_digits: int=4) -> None:
        self.model = model
        self.img_path= img_path
        self.n_digits=n_digits
        self.images_load= self.image_loader()
        
    
    

    def image_loader(self):
        
        
        mat = cv2.imread(self.img_path)
        rgb_img=cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        gray=mat.copy()
        gray_digits = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        return  gray_digits


    
    def preprocess(self)->list:
        
        
        _, threshold = cv2.threshold(self.images_load, 100,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        
        dilated = cv2.dilate(threshold, (3, 3), iterations=5)
        
        # find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        plt.imshow(sorted_contours)
        plt.show()
        final_contours = sorted_contours[:self.n_digits]

       
        cropped_digits = []
        for i in range(len(final_contours)):
            x, y, w, h = cv2.boundingRect(final_contours[i])

            x = x - 100
            y = y - 100
            w = w + 200
            h = w + 200

            resized_digit = cv2.resize(
                dilated[y:y+h, x:x+w], (28, 28), interpolation=cv2.INTER_CUBIC)
            cropped_digits.append(resized_digit)

        return cropped_digits




    def predict(self,mat):
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.model.eval()
        with torch.no_grad():
            trsf_digit = transform(mat).unsqueeze(0)
            logsoft = self.model(trsf_digit)
            return torch.argmax(logsoft, 1)


    def show_digits(self):
        string_digits = ''
        for digit in self.preprocess():
            string_digits += str(self.predict(digit).item())
        print(string_digits)



if __name__=='__main__':
    numbers=NumberPredictor(model, img_path='digits/codenum.jpeg')
    numbers.show_digits()

mat=cv2.imread('digits/codenum.jpeg')

print(type(mat))