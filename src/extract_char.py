import cv2
import numpy as np
import os
import random as rng
from matplotlib import pyplot as plt

def extract_char(threshed_img,org_img):

    contours, _ = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    org_img = cv2.cvtColor(org_img,cv2.COLOR_GRAY2BGR)
    drawn_img = draw_contours(contours, boundRect, contours_poly,org_img) 
    return drawn_img

def draw_contours(contours, boundRect, contours_poly, org_img):    
        
        #drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) >= 100 and cv2.contourArea(contours[i])<= 800:#object area tresholds
                if boundRect[i][2]<2*(boundRect[i][3]):
                    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                    #cv2.drawContours(drawing, contours_poly, i, color) #To draw contours
                    img2 = cv2.rectangle(org_img, (int(boundRect[i][0]), int(boundRect[i][1])), \
                      (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), \
                      color, 2)
                    
        return org_img

def main():
    input_path = "/home/konan/Desktop/plate/img_plate"
    image_names = os.listdir(input_path)
    print(" Program started....","\n\n","Press any key for next image", "\n","Press q to quit")

    for image_name in image_names:
        if image_name[-4:] == ".png" or image_name[-4:] == ".jpg":
            img = cv2.imread(input_path + "/" + image_name,0)
            cv2.imshow("org", img)
            threshed_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,55,5)
            drawn_img = extract_char(threshed_img,img)
            cv2.imshow("drawn_img", drawn_img)
            key = cv2.waitKey(0)
            
            if key & 0xFF == ord('q'):
                print(" Shut down!")
                break
                
if __name__ == "__main__":
    main()  
