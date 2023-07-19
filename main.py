import cv2
import sys
from model import ImageStitching


if __name__ == "__main__":
    model = ImageStitching()

    img1_path = "1.jpg"
    img2_path = "2.jpg"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    final = model.blending(img1, img2)
    cv2.imwrite("panorama.jpg", final)
