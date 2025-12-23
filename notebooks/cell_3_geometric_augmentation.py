# Cell 3: Geometric Augmentations (Histology-specific)
import torchvision.transforms as transforms
import torch.nn.functional as F

class HistologyAugmentation:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transforms = transforms.Compose([
            transforms.RandomRotation(degrees=[0, 90, 180, 270]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ])
        
    def __call__(self, image):
        return self.transforms(image)

class ElasticTransform:
    def __init__(self, alpha=1, sigma=50, alpha_affine=50):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        
    def __call__(self, image):
        if random.random() > 0.3:
            return image
            
        if isinstance(image, torch.Tensor):
            was_tensor = True
            image_np = image.permute(1, 2, 0).numpy()
        else:
            was_tensor = False
            image_np = np.array(image)
            
        shape = image_np.shape
        shape_size = shape[:2]
        
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, 
                          [center_square[0]+square_size, center_square[1]-square_size], 
                          center_square - square_size])
        pts2 = pts1 + np.random.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        
        image_np = cv2.warpAffine(image_np, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        if was_tensor:
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            image = image_np
            
        return image

histology_aug = HistologyAugmentation()
elastic_transform = ElasticTransform()
print("Geometric augmentation transforms initialized")