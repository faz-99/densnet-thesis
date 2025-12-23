# Cell 8: Complete Preprocessing Pipeline Integration
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class BreakHisPreprocessingPipeline:
    def __init__(self, config=PREPROCESSING_CONFIG, img_size=224):
        self.config = config
        self.img_size = img_size
        
        if config['stain_normalization']:
            self.stain_normalizer = StainNormalizer(method=config['stain_method'])
        
        if config['image_enhancement']:
            self.image_enhancer = ImageEnhancer()
            
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
        
    def _build_train_transforms(self):
        transforms_list = []
        
        transforms_list.extend([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size))
        ])
        
        if self.config['geometric_augmentation']:
            transforms_list.extend([
                transforms.RandomRotation(degrees=[0, 90, 180, 270]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0))
            ])
        
        if self.config['color_augmentation']:
            transforms_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.15)
            )
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.config['advanced_augmentation']:
            transforms_list.append(RandomErasing(probability=0.15))
            
        return transforms.Compose(transforms_list)
    
    def _build_val_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image, is_training=True):
        """Complete preprocessing pipeline for a single image"""
        if self.config['image_enhancement']:
            image = self.image_enhancer.enhance_image(image)
        
        if self.config['stain_normalization']:
            image = self.stain_normalizer.normalize(image)
        
        if is_training:
            return self.train_transforms(image)
        else:
            return self.val_transforms(image)
    
    def create_data_loader(self, dataset, batch_size=32, is_training=True, class_balancing=True):
        """Create data loader with optional class balancing"""
        if is_training and class_balancing and self.config['class_balancing']:
            labels = [dataset[i][1] for i in range(len(dataset))]
            balancer = ClassBalancer(labels)
            sampler = balancer.create_balanced_sampler()
            
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=is_training,
                num_workers=4,
                pin_memory=True
            )

preprocessing_pipeline = BreakHisPreprocessingPipeline(config=PREPROCESSING_CONFIG)
print("Complete preprocessing pipeline initialized")
print(f"Configuration: {PREPROCESSING_CONFIG}")