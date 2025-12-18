import cv2
import torchvision.transforms as transforms
import torch
import json
import os
from PIL import Image

def preprocess(ori_img, ori_boxes, device):
    h_rsz,w_rsz = 512,512
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    h_orig,w_orig = ori_img.shape[:2]
    rsz_image = cv2.resize(ori_img, (h_rsz, w_rsz))
    h_scale, w_scale = h_rsz / h_orig, w_rsz / w_orig
    if len(ori_boxes) >3:
         ori_boxes = ori_boxes[:3]
    for i in range(3-len(ori_boxes)):
            ori_boxes.append(ori_boxes[i])
    rsz_boxes = []
    for box in ori_boxes:
        y_tl, x_tl, y_br, x_br = box
        y_tl = int(y_tl * h_scale)
        y_br = int(y_br * h_scale)
        x_tl = int(x_tl * w_scale)
        x_br = int(x_br * w_scale)
        rsz_boxes.append([y_tl, x_tl, y_br, x_br])
    rsz_image = transforms.ToTensor()(rsz_image)
    rsz_boxes = torch.tensor(rsz_boxes, dtype=torch.float64).unsqueeze(0).to(device)
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rsz_image = normalize_fn(rsz_image).unsqueeze(0).to(device)
    return rsz_image, rsz_boxes

# def process_exemplars(exemplar_json_path, device):
#     # Load the exemplars from the JSON file
#     with open(exemplar_json_path, 'r') as f:
#         exemplars = [json.loads(line) for line in f]

#     # Prepare to collect the support tensors
#     combined_rsz_boxes = []
    
#     # Process each exemplar
#     for exemplar in exemplars:
#         exemplar_image_path = os.path.join('./data/CARPK/Images/', exemplar['filename'])
#         exemplar_img = cv2.imread(exemplar_image_path)
#         ori_box = [list(map(int, exemplar['box']))]  # Convert box from JSON

#         # Preprocess each exemplar image and box
#         _, rsz_boxes = preprocess(exemplar_img, ori_box, device)

#         # Collect the resized boxes
#         combined_rsz_boxes.append(rsz_boxes)

#     # Combine all the support tensors to have the desired shape of [1, 3, 4]
#     # We will stack them and take the mean, or we can pick any logic that fits the scenario.
#     # For simplicity, let's assume we will concatenate them along dim=1 and ensure final size is [1, 3, 4]
#     if len(combined_rsz_boxes) >= 3:
#         combined_rsz_boxes = torch.cat(combined_rsz_boxes[:3], dim=1)  # Combine first 3 supports
#     else:
#         # If there are fewer than 3 boxes, repeat the first one to make up the size
#         while len(combined_rsz_boxes) < 3:
#             combined_rsz_boxes.append(combined_rsz_boxes[0])

#         combined_rsz_boxes = torch.cat(combined_rsz_boxes, dim=1)

#     # Ensure final shape is [1, 3, 4]
#     combined_rsz_boxes = combined_rsz_boxes[:, :3, :4]  # Truncate if needed

#     return combined_rsz_boxes

# def supp(exemplar_json_path, device):
#     """
#     This function loads exemplars from exemplar_json_path, processes them, and returns 
#     support images and support bounding boxes for further use in object counting models.
    
#     Arguments:
#     - exemplar_json_path: Path to the exemplars JSON file.
#     - device: Device (cpu or cuda) to use for tensor processing.
    
#     Returns:
#     - support_images: A list of tensors representing the processed support images.
#     - support_boxes: A tensor of bounding boxes for each support image.
#     """
#     with open(exemplar_json_path, 'r') as f:
#         exemplars = [json.loads(line) for line in f]

#     support_images = []  # List to hold the processed support images
#     support_boxes = []   # List to hold the corresponding bounding boxes

#     # Image preprocessing (resize and normalization)
#     preprocess_transform = transforms.Compose([
#         transforms.Resize((512, 512)),  # Resizing image (you can change the size if needed)
#         transforms.ToTensor(),          # Convert to Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
#     ])

#     # Iterate through each exemplar in the json file
#     for exemplar in exemplars:
#         image_path = exemplar['filename']  # Get the image path from the exemplar
#         box = exemplar['box']              # Get the bounding box from the exemplar

#         # Load the image from file
#         img = Image.open(os.path.join('./data/CARPK/Images', image_path)).convert("RGB")
        
#         # Apply the preprocessing transform to the image
#         img_tensor = preprocess_transform(img).to(device)
        
#         # Append the processed image and its bounding box to the respective lists
#         support_images.append(img_tensor)
#         support_boxes.append(box)  # This assumes box is a list of coordinates like [x_min, y_min, x_max, y_max]

#     # Convert support_boxes to a tensor if needed (for compatibility with models)
#     support_boxes_tensor = torch.Tensor(support_boxes).to(device)

#     return support_images, support_boxes_tensor