import torch
from PIL import Image
from torchvision import transforms

from config import *
import timm

from model.jiangzao import *

# Load the input image
input_image = Image.open('C:\\zlg\\ISIC_0024470.jpg')

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# transform_valid = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.CenterCrop(img_size),
#     transforms.ToTensor(),
#     transforms.Normalize(dataset_mean, dataset_std)
# ])

# Apply the transformation pipeline to the input image
input_tensor = transform(input_image).unsqueeze(0)

# Instantiate the model and load the trained weights

model = timm.models.convnext.convnext_base(pretrained=True, num_classes=class_num).to(device)
jiangzaomodel = jiangzaonet(channels=3)
model = zong(jiangzaomodel, model)
# a = torch.load('../weight/save/convnext_no.pth')
# print(a.keys())
# exit()
model.load_state_dict(torch.load('../weight/save/convnext_no.pth')['model_state_dict'])

# Reconstruct the image
print(input_tensor.shape)
jiangzaonet = model.jiangzao
output_tensor = jiangzaonet(input_tensor.cuda())

# Convert the output tensor to a numpy array
output_array = output_tensor.squeeze(0).detach().cpu().numpy()

output_array = output_array.transpose((1, 2, 0))

# Convert the numpy array to a PIL image
output_image = Image.fromarray((output_array * 255).astype('uint8'))

# Display or save the output image
output_image.show()
output_image.save('output_image.jpg')
