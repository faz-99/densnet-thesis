# Cell 7: Image Quality Enhancement (CLAHE, Background Removal)
import cv2
import numpy as np
from skimage import morphology
from scipy import ndimage

class ImageEnhancer:
    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = self.clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = self.clahe.apply(image)
        return enhanced
    
    def remove_background(self, image, threshold=230):
        """Remove white background and artifacts"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        tissue_mask = gray < threshold
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=1000)
        tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=1000)
        
        kernel = np.ones((5,5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        
        if len(image.shape) == 3:
            result = image.copy()
            result[~tissue_mask] = [255, 255, 255]
        else:
            result = image.copy()
            result[~tissue_mask] = 255
            
        return result, tissue_mask
    
    def enhance_image(self, image):
        """Complete image enhancement pipeline"""
        enhanced, mask = self.remove_background(image)
        enhanced = self.apply_clahe(enhanced)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced

class BackgroundRemover:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def remove_white_background(self, image):
        """Remove white background using color thresholding"""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        tissue_mask = cv2.bitwise_not(white_mask)
        
        result = image.copy()
        result[white_mask > 0] = [255, 255, 255]
        
        return result

image_enhancer = ImageEnhancer()
background_remover = BackgroundRemover()
print("Image quality enhancement tools initialized")