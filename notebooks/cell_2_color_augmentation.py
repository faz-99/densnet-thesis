# Cell 2: Color Augmentation for H&E Staining Variations
import torchvision.transforms as transforms
import torch
import random

class ColorAugmentation:
    def __init__(self, hue_range=0.15, saturation_range=0.2, brightness_range=0.2, contrast_range=0.2):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            saturation=saturation_range,
            hue=hue_range
        )
        
    def __call__(self, image):
        if random.random() > 0.5:
            return self.color_jitter(image)
        return image

class HSVAugmentation:
    def __init__(self, hue_shift=15, sat_shift=20, val_shift=20):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] += random.uniform(-self.hue_shift, self.hue_shift)
        hsv[:,:,1] *= random.uniform(1-self.sat_shift/100, 1+self.sat_shift/100)
        hsv[:,:,2] *= random.uniform(1-self.val_shift/100, 1+self.val_shift/100)
        hsv[:,:,0] = np.clip(hsv[:,:,0], 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return transforms.ToTensor()(rgb)

color_aug = ColorAugmentation()
hsv_aug = HSVAugmentation()
print("Color augmentation transforms initialized")