import numpy as np
import cv2


class Tools:
    
    @staticmethod
    def change_color_fuzzycmeans(cluster_membership, clusters):
        img = []
        for pix in cluster_membership.T:
            img.append(clusters[np.argmax(pix)])
        return img

    @staticmethod
    def bwarea(img):
        row = img.shape[0]
        col = img.shape[1]
        total = 0.0
        for r in range(row-1):
            for c in range(col-1):
                sub_total = img[r:r+2, c:c+2].mean()
                if sub_total == 255:
                    total += 1
                elif sub_total == (255.0/3.0):
                    total += (7.0/8.0)
                elif sub_total == (255.0/4.0):
                    total += 0.25
                elif sub_total == 0:
                    total += 0
                else:
                    r1c1 = img[r,c]
                    r1c2 = img[r,c+1]
                    r2c1 = img[r+1,c]
                    r2c2 = img[r+1,c+1]
                    
                    if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                        total += 0.75
                    else:
                        total += 0.5
        return total
    
    @staticmethod            
    def imclearborder(imgBW):

        # Given a black and white image, first find all of its contours
        radius = 2
        imgBWcopy = imgBW.copy()
        image, contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE)

        # Get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]    

        contourList = [] # ID list of contours that touch the border

        # For each contour...
        for idx in np.arange(len(contours)):
            # Get the i'th contour
            cnt = contours[idx]

            # Look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]

                # If this is within the radius of the border
                # this contour goes bye bye!
                check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
                check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

                if check1 or check2:
                    contourList.append(idx)
                    break

        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

        return imgBWcopy

    #### bwareaopen definition
    @staticmethod
    def bwareaopen(imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
        imgBWcopy = imgBW.copy()
        image, contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, determine its total occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if (area >= 0 and area <= areaPixels):
                cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

        return imgBWcopy      

    @staticmethod
    def imfill(im_th):
        
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        
        return im_out