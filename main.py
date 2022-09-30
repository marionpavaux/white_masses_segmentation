"""
Main code for the white masses segmentation project 
Image Processing Toolbox at 'Les Mines de Saint-Etienne'
Deadline : 22/05/21
"""

import numpy as np 
import cv2 
from scipy import ndimage
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

import time
from typing import Tuple

# Load the video 
VIDEO = cv2.VideoCapture('sluglines-master/V_20200630_112727_N0.mp4')
THRESHOLD = 120
WEIGHT_INI = 1920
HEIGHT_INI = 1080


def filtering(image: np.ndarray): 
    imconv = ndimage.median_filter(image,(3,3))
    return imconv


def thresholding(image: np.ndarray, threshold: int) -> np.ndarray: 
    '''
    Returns the initial image with a modification in the pixels. 
    If the pixel is above threshold it becomes white and otherwise black
    '''
    thresholded_image = np.where(image>=threshold, 255, 0)
    return thresholded_image 


def counting(image: np.ndarray) -> Tuple[list, list]: 
    """
    Returns the number of white pixels directly following each other for each column in the image
    and return the position of the lowest white pixel for each column in the image
    """
    counting = []
    height = []
    size = image.shape
    low, up = 0, 0

    for column in range(size[1]):
        # begin at the bottom 
        raw = size[0]-1
        while image[raw, column] != 255 and raw >= 0:
            raw -= 1
        if image[raw, column] == 255:  
            low = raw 
            # while pixels are still white
            while image[raw][column] == 255 and raw >= 0:
                raw -= 1
            # if pixel becomes black
            if image[raw][column] == 0:
                up = raw 

        counting.append(low-up)
        height.append(low)

    return counting, height

     
def compute_features(thickness: list, height: list, image: np.ndarray)->dict:
    '''
    Returns a dictionaties containing features for each slug : 
    the first and last y-position of the slug, the mean thickness and mean height
    '''
    size  = image.shape
    # slugs maximum thickness
    max_thickness = np.max(thickness)
    sigma = np.std(thickness)
    
    # will contain a list for each slug with the beginning of a slug, end of a slug, mean thickness and mean height
    slugs_features = {'begin' : [], 'end' : [], 'thickness' : [], 'height': []}
    column = 0
   
    while column<size[1]: 
        # slug detection
        if max_thickness - 2*sigma <= thickness[column]:
            # adding 75 in width as we croped the image according to the region of interest
            slugs_features['begin'].append(column+75)
            slug_thickness = [thickness[column]]
            # adding 600 in height as we croped the image according to the region of interest
            slug_height = [height[column] + 600]
            
            while max_thickness - 2*sigma <= thickness[column] and column<size[1]-1 :   
                column += 1
                slug_thickness.append(thickness[column])
                # adding 600 in height as we croped the image according to the region of interest
                slug_height.append(height[column]+ 600)

            # adding 75 in width as we croped the image according to the region of interest
            slugs_features['end'].append(column+75)
            slugs_features['thickness'].append(np.mean(slug_thickness))
            slugs_features['height'].append(np.mean(slug_height))
        
        column +=1 
    return slugs_features
    

def color_slug(image: np.ndarray,slugs_features: dict): 
    '''
    Returns the image with slugs colored in green
    '''
    for slug in range(len(slugs_features['begin'])):
        
        width_to_color = np.arange(slugs_features['begin'][slug] , slugs_features['end'][slug])
        height_to_color = np.arange(int(slugs_features['height'][slug] - 8), int(slugs_features['height'][slug]))
        
        for height in height_to_color:
            # color in green
            image[height,width_to_color,0] = 0
            image[height,width_to_color,1] = 255
            image[height,width_to_color,2] = 0
    return image
    

def display_feature(feature: np.ndarray, feature_title: str, image: np.ndarray): 
    '''
    Display the original image and a corresponding feature plot
    '''
    fig, axs = plt.subplots(2,1)

    # display original image
    axs[0].imshow(image, cmap="gray") 
    axs[0].set_title("Original image")
    
    # display feature 
    axs[1].plot(feature)
    axs[1].set_title(feature_title)
    
    fig.savefig(f'{feature_title}.png', transparent=True)
    plt.close(fig)
        

if __name__ == "__main__":

    begin_time = time.time()

    colored_frames = []
    thickness_over_time=[]
    height_over_time = []
    slugs_features_over_time=[]

    ret = True 

    read = 0
    num = 0
    print("Image Processing\nThis won't take long...\n")
    while ret == True : 
        ret, frame_ini = VIDEO.read() 
        num += 1
        # video has 30 frames per second, beginning after 20 seconds
        if num >= 600: 
            if read == 0:
                # Crop image and taking only grey levels
                frame = frame_ini[600:850,75:1700,0]
                frame = filtering(frame)
                frame = thresholding(frame,THRESHOLD)
                count, height = counting(frame)
                
                # computing features on the positions, thickness and height of the slugs
                slugs_features = compute_features(count,height,frame)
                slugs_features_over_time.append(slugs_features)
             
                # computing mean thickness and height value of all slugs on one frame
                thickness_over_time.append(np.mean(slugs_features['thickness']))
                height_over_time.append(np.mean(slugs_features['height']))
                
                # coloring slugs in green
                colored_frame = color_slug(frame_ini,slugs_features)
                colored_frames.append(colored_frame)
            
            # moving with a step of 5 images
            if read == 5: 
                read = 0
            else :
                read += 1
               
    mean_thickness = np.mean(thickness_over_time)
    print(f"The mean slug thickness over time is: {mean_thickness}") 
    
    mean_height = np.mean(height_over_time)
    print(f"The mean slug height over time is: {mean_height}")
    
    height, width = colored_frames[0].shape[0], colored_frames[0].shape[1]
    out = cv2.VideoWriter("green_masses.avi",cv2.VideoWriter_fourcc(*'DIVX'),6,(width,height))
    print("Writing the video...")

    for frame in range(len(colored_frames)):
        out.write(np.uint8(colored_frames[frame])) 
  
    VIDEO.release() 
    out.release()

    end_time = time.time()
    print(f"\nTime elapsed: {end_time - begin_time}")