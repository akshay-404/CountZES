import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
from tqdm import tqdm
import cv2
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import warnings
warnings.filterwarnings('ignore')
from transformers import CLIPProcessor, CLIPModel
import math
from scipy.stats import entropy, gaussian_kde
from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from DSALVANet.utils.fsol_modules import vis_DSALVANet
from DSALVANet.utils.data_preprocess import preprocess
from DSALVANet.utils.model_helper import build_model
from DSALVANet.utils.PerSense_countr import vis_countr
from GroundingDINO.groundingdino.util.inference import Model
from typing import List
from CounTR import models_mae_cross
from CounTR.demo import load_image, run_one_image
from skimage.feature import peak_local_max
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='countzes_mbm')
    parser.add_argument('--ckpt', type=str, default='./data/sam_vit_b_01ec64.pth')
    parser.add_argument('--sam_type', type=str, default='vit_b')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--visualize', type=bool, default= False) # Change to True for visualization
    parser.add_argument('--fsoc', type=str, default='countr') #use countr for COUNTR BMVC 22   
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    print("Args:", args)

    print("======> Load SAM" )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'data/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
    elif args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', 'data/sam_vit_b_01ec64.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
    elif args.sam_type == 'vit_l':
        sam_type, sam_ckpt = 'vit_l', 'data/sam_vit_l_0b3195.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
    else:
        raise ValueError(f"Unsupported sam_type '{args.sam_type}'. Choose from 'vit_b', 'vit_h', or 'vit_t'.")
    sam.eval()
    print("======> Done" )  

    print("======> Load Grounding Detector" )
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print("======> Done" )

    print("======> Load Object Counter" )

    if args.fsoc == 'countr':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_countr = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
            model_countr.to(device)
            model_without_ddp = model_countr

            checkpoint = torch.load('./CounTR/output_allnew_dir/FSC147.pth', map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            

            model_countr.eval()
            counter_model = model_countr

    elif args.fsoc == 'DSALVANet':

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser_input = argparse.ArgumentParser(description="Test code of DSALVANet")
        parser_input.add_argument("-w", "--weight", type=str, default="./DSALVANet/checkpoints/checkpoint_200.pth", help="Path of weight.")
        parser_input.add_argument('--visualize', type=bool, default= False)
        parser_input.add_argument('--fsoc', type=str, default='DSALVANet') #use countr for COUNTR BMVC 22

        args_counter = parser_input.parse_args()
        weight_path = args_counter.weight
        counter_model = build_model(weight_path,device)
        print("======> Done" )

    print("======> Load CLIP" )

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("======> Done" )

    if not hasattr(nn, "RMSNorm"):
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-8):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(normalized_shape))

            def forward(self, x):
                norm_x = x.norm(2, dim=-1, keepdim=True)
                rms = norm_x * (1.0 / (x.size(-1) ** 0.5))
                return self.weight * x / (rms + self.eps)
        nn.RMSNorm = RMSNorm
    anyup = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()

    images_path = "./data/mbm/Images/"
    masks_path = args.data + '/mbm/Images/'
    output_path = './outputs/' + args.outdir
    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')

    count_item = 0
    mae = 0
    mse = 0
    infer_time = 0

    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            countzes(args, obj_name, images_path, masks_path, output_path, sam, grounding_dino_model, counter_model, infer_time, clip_model, clip_processor, count_item, mae, mse, anyup)



def countzes(args, obj_name, images_path, masks_path, output_path, sam, grounding_dino_model, counter_model, infer_time, clip_model, clip_processor, count_item, mae, mse, anyup):
    
    test_images_path = os.path.join(images_path, obj_name)
    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    

    print('======> Start Counting')
    loop_over = len(os.listdir(test_images_path))
    for test_idx in tqdm(range(loop_over//2)):

        image_name = f"{test_idx:02}.png" if test_idx < 10 else f"{test_idx}.png"
        output_file = os.path.join(output_path, image_name)

        # Load test image
        test_idx = '%02d' % test_idx
        img_file = test_idx + '.png'
        test_image_path = test_images_path + '/' + test_idx + '.png'
        filename = img_file
        anno_image_path = os.path.join(test_images_path, f"{test_idx}A.png")
        anno_img = cv2.imread(anno_image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(anno_img, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        gt_count = num_labels - 1 

        for name, param in sam.named_parameters():
            param.requires_grad = False
        predictor = SamPredictor(sam)     
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)


        class1 = [obj_name]

        # ==================================================
        # Stage 1: Detection Anchored Exemplar (DAE)
        # ==================================================     
        sim_clip = compute_clip_similarity_map(test_image, class1[0],
                                      clip_model, clip_processor, device)        
        sim_clip = normalize_sim_map(sim_clip)      
        image_height, image_width, _ = test_image.shape  

        SOURCE_IMAGE_PATH = test_image_path
        print(f"Target object: {class1}")
        CLASSES = class1        
        BOX_TRESHOLD = 0.20
        TEXT_TRESHOLD = 0.15
        image = cv2.imread(SOURCE_IMAGE_PATH)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes = CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        if detections is None or len(detections.xyxy) == 0:
            print("No detections found. Retrying with fallback class ['object']...")
            CLASSES = ['object']
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=CLASSES,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
        entropies = [compute_clip_entropy(sim_clip, box) for box in detections.xyxy]
        confidences = detections.confidence.tolist()
        valid_indices = [i for i, e in enumerate(entropies) if np.isfinite(e)]
        if len(valid_indices) == 0:
            raise ValueError("All boxes had invalid entropies.")

        valid_entropies = [entropies[i] for i in valid_indices]
        valid_confidences = [confidences[i] for i in valid_indices]

        # Initial Box Selection ---
        norm_entropy = normalize(valid_entropies)
        valid_confidences_arr = np.array(valid_confidences)
        alpha = 0.5  
        scores = [alpha * c + (1 - alpha) * (1 - e) for c, e in zip(valid_confidences_arr, norm_entropy)]
        best_relative_index = int(np.argmax(scores))
        best_index = valid_indices[best_relative_index]
        bbox_coord = detections.xyxy[best_index]

        
        #Similarity Guided SAM-Based Exemplar Selection (SSES)
        dae, best_mask, best_entropy = SSES(   
            test_image,
            bbox_coord,
            predictor,
            clip_model, clip_processor, class1[0],
            device,
            sim_clip=sim_clip,
            num_points=16,
            min_distance=5,
            start_percentile=80,
            min_percentile=50,
            step=10,
            filename=filename,
        )

        if dae is not None:
            bbox_coord = dae
        H, W = test_image.shape[:2]    
        x0, y0, x1, y1 = map(int, bbox_coord)
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H, y1))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)


        exemplar = test_image[y0:y1, x0:x1]
        predictor.set_image(exemplar)             
        ex_feat = predictor.features.squeeze(0)
        ex_emb = ex_feat.view(ex_feat.shape[0], -1).mean(1) 
        ex_emb = ex_emb / ex_emb.norm()      


        predictor.set_image(test_image)
        full_feat = predictor.features.squeeze(0)     # (C, h, w)
        full_feat = full_feat / full_feat.norm(dim=0, keepdim=True)  # (C, h, w)
        kernel = ex_emb.view(1, -1, 1, 1)                # (1, C, 1, 1)
        feat = full_feat.unsqueeze(0)                    # (1, C, h, w)
        sim_map = F.conv2d(feat, kernel)                 # (1,1,h,w)
        sim_map = sim_map.squeeze(0).squeeze(0)          # (h, w)
        sim_lowres = sim_map.unsqueeze(0).unsqueeze(0)  # (1,1,h_feat,w_feat)


        sim_up = predictor.model.postprocess_masks(
            sim_lowres,
            input_size=predictor.input_size,
            original_size=predictor.original_size
        ).squeeze(0).squeeze(0)  # (H, W)
        sim_up = sim_up.detach().cpu().numpy()
        sim_up = normalize_sim_map(sim_up)
        
        if len(detections.xyxy) == 1:
               
            dae, best_mask, best_entropy = SSES(
                test_image,
                bbox_coord,
                predictor,
                clip_model, clip_processor, class1[0],
                device,
                sim_clip=sim_up,
                num_points=4,
                min_distance=5,
                start_percentile=90,
                min_percentile=80,
                step=10,
                filename=filename,
            )

            if dae is not None:
                bbox_coord = dae

            x0, y0, x1, y1 = map(int, bbox_coord)   


        # ==================================================
        # Stage 2: Density Guided Exemplar (DGE)
        # ==================================================
        ori_boxes = [[y0, x0, y1, x1]]
        src_img = cv2.imread(test_image_path)
        query, supports = preprocess(src_img, ori_boxes,device)

        if args.fsoc == 'countr': 
            samples, boxes,orig_boxes, pos, W, H, new_W, new_H = load_image(test_image_path, ori_boxes)
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, elapsed_time, density_pred, density_raw = run_one_image(samples, boxes, pos, counter_model, W, H, filename, new_W, new_H)
            vis_output, count, density_pred = vis_countr(src_img, orig_boxes, result, filename, density_pred)

        elif args.fsoc == 'DSALVANet':
            output = counter_model(query,supports)
            vis_output, count, density_pred, density_raw = vis_DSALVANet(src_img,ori_boxes,output, filename)

        mean_gray = density_pred.mean()
        std_gray = density_pred.std()
        threshold = mean_gray + 2*std_gray

        #P2P Prompting
        peaks = peak_local_max(density_pred, min_distance=5, threshold_abs=threshold)
        point_prompts = filter_peak_prompts(peaks, detections)
        mask_list, box_list_sam = get_masks_and_boxes(predictor, image, point_prompts)
        box_list_det = [[int(x0), int(y0), int(x1), int(y1)] for (x0, y0, x1, y1) in detections.xyxy]
        box_list = box_list_sam + box_list_det
        box_list.append(dae)

        #Single Instance Filtering via RoI Count
        def box_count_from_density(box, density_map):
            x0, y0, x1, y1 = map(int, box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(density_map.shape[1]-1, x1), min(density_map.shape[0]-1, y1)
            return float(density_map[y0:y1+1, x0:x1+1].sum())
        counts_per_box = [box_count_from_density(b, density_raw) for b in box_list]
        single_instance_boxes1 = [b for b, c in zip(box_list, counts_per_box) if 1 < c < 2]
        if len(single_instance_boxes1) == 0:
            print("[Warning] No single-instance boxes found (count<=1). Falling back to all boxes.")
            single_instance_boxes1 = box_list


        #Pseudo GT Guided Exemplar Selection (GGES)
        sam_box_count = []
        src_img = cv2.imread(test_image_path)
        arg_tuples = [
            (det_box, test_image_path, counter_model, args, src_img, filename, device)
            for det_box in single_instance_boxes1
        ]
        with ThreadPoolExecutor(max_workers=12) as executor:
            sam_box_count = list(executor.map(process_box, arg_tuples))
        pseudo_gt_result = cluster_counts(sam_box_count, single_instance_boxes1, filename)
        pseudo_gt_count = pseudo_gt_result["kde_pseudo_gt"]
        entropies_all = [compute_clip_entropy(sim_up, box) for box in single_instance_boxes1]
        entropies_all = [e if np.isfinite(e) else 1.0 for e in entropies_all]  # guard NaNs
        norm_entropy = normalize(entropies_all)
        diffs = [abs(c - pseudo_gt_count) for c in sam_box_count]
        dmin, dmax = min(diffs), max(diffs)
        if dmax == dmin:
            closeness_norm = [1.0] * len(diffs)
        else:
            closeness_norm = [1.0 - ((d - dmin) / (dmax - dmin)) for d in diffs]
        alpha = 0.5  
        scores = [alpha * c_close + (1 - alpha) * (1 - e)
                for c_close, e in zip(closeness_norm, norm_entropy)]
        best_idx = int(np.argmax(scores))
        dge = single_instance_boxes1[best_idx]

        # ==================================================
        # Stage 3: Feature Consensus Exemplar (FCE)
        # ==================================================
        x0, y0, x1, y1 = map(int, dge)
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H, y1))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
     
        dge_box = [[y0, x0, y1, x1]]
        exemplar = test_image[y0:y1, x0:x1]
        predictor.set_image(exemplar)
        ex_feat = predictor.features.squeeze(0)
        ex_emb = ex_feat.view(ex_feat.shape[0], -1).mean(1)
        ex_emb = ex_emb / ex_emb.norm()
        predictor.set_image(test_image)
        full_feat = predictor.features.squeeze(0)
        full_feat = full_feat / full_feat.norm(dim=0, keepdim=True)
        kernel = ex_emb.view(1, -1, 1, 1)
        feat = full_feat.unsqueeze(0)
        sim_map = F.conv2d(feat, kernel).squeeze(0).squeeze(0)
        sim_lowres = sim_map.unsqueeze(0).unsqueeze(0)
        sim_up = predictor.model.postprocess_masks(
            sim_lowres,
            input_size=predictor.input_size,
            original_size=predictor.original_size
        ).squeeze(0).squeeze(0).detach().cpu().numpy()

        sim_up = normalize_sim_map(sim_up)

        #Density Map leveraging dge
        query, supports = preprocess(test_image, dge_box, device)
        if args.fsoc == 'countr':
            samples, boxes,orig_boxes, pos, W, H, new_W, new_H = load_image(test_image_path, dge_box)
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, elapsed_time, density_pred, density_raw = run_one_image(samples, boxes, pos, counter_model, W, H, filename, new_W, new_H)
            vis_output, count, density_pred = vis_countr(src_img, orig_boxes, result, filename, density_pred)
        elif args.fsoc == 'DSALVANet':
            output = counter_model(query,supports)
            vis_output, count, density_pred, density_raw = vis_DSALVANet(src_img,dge_box,output, filename)


        mean_gray = density_pred.mean()
        std_gray = density_pred.std()
        threshold = mean_gray + 2*std_gray

        #P2P prompting
        peaks2 = peak_local_max(density_pred, min_distance=5, threshold_abs=threshold)
        point_prompts2 = filter_peak_prompts(peaks2, detections)
        dark_img = (src_img * 0.4).astype(np.uint8)
        mask_list_new, box_list_sam_new = get_masks_and_boxes(predictor, image, point_prompts2)
        for box in box_list_sam_new:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(dark_img, (x0, y0), (x1, y1), color=(0, 165, 255), thickness=2)  # Orange
        box_list_sam_new.append(dae)
        box_list_sam_new.append(dge)

        if len(box_list_sam_new) == 0:
            print ("No SAM boxes to cluster.")
            box_list_sam_new = [dge]
             
        def contains(box_a, box_b):
            ax0, ay0, ax1, ay1 = box_a
            bx0, by0, bx1, by1 = box_b
            return ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1
        filtered_boxes = []
        for i, box_a in enumerate(box_list_sam_new):
            is_container = False
            for j, box_b in enumerate(box_list_sam_new):
                if i != j and contains(box_a, box_b):
                    is_container = True
                    break
            if not is_container:
                filtered_boxes.append(box_a)


        if len(filtered_boxes) == 0:
            print("[Warning] All boxes were filtered as containers. Falling back to original list.")
            filtered_boxes = box_list_sam_new

        #Single Instance Filtering via RoI Count
        counts_per_box = [box_count_from_density(b, density_raw) for b in filtered_boxes]
        single_instance_boxes = [b for b, c in zip(filtered_boxes, counts_per_box) if 1 < c < 2]
        if len(single_instance_boxes) == 0:
            print("[Warning] No single-instance boxes found (count<=1). Falling back to all boxes.")
            single_instance_boxes = filtered_boxes

        #Feature-based Representative Exemplar Selection (FRES)
        _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        def to_imagenet_tensor(img_rgb_uint8: np.ndarray, device):
            assert img_rgb_uint8.ndim == 3 and img_rgb_uint8.shape[2] == 3
            t = torch.from_numpy(img_rgb_uint8).float().permute(2,0,1).unsqueeze(0) / 255.0  # (1,3,H,W)
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            return t.to(device)

        predictor.set_image(test_image)
        full_feat = predictor.features.squeeze(0)  # (C, h_feat, w_feat)
        C, h_feat, w_feat = full_feat.shape
        H, W = test_image.shape[:2]      
        hr_image = to_imagenet_tensor(test_image, device)
        if not isinstance(full_feat, torch.Tensor):
            full_feat = torch.from_numpy(full_feat)
        lr_features = full_feat.unsqueeze(0).to(device).contiguous()
        with torch.no_grad():
            hr_features = anyup(hr_image, lr_features, q_chunk_size=256, output_size=(256, 256)) #Feature Upsampling
        hr_features = hr_features.squeeze(0).contiguous()     # (C, H, W)

        def box_to_featvec(box_xyxy):
            x0, y0, x1, y1 = map(int, box_xyxy)
            # clamp to image bounds
            x0 = max(0, min(256 - 1, x0)); x1 = max(0, min(256, x1))
            y0 = max(0, min(256 - 1, y0)); y1 = max(0, min(256, y1))
            if x1 <= x0: x1 = min(W, x0 + 1)
            if y1 <= y0: y1 = min(H, y0 + 1)

            # map to feature-grid coords
            fx0 = int(round(x0 * (w_feat / float(256))))
            fx1 = int(round(x1 * (w_feat / float(256))))
            fy0 = int(round(y0 * (h_feat / float(256))))
            fy1 = int(round(y1 * (h_feat / float(256))))
            # clamp
            fx0 = max(0, min(w_feat - 1, fx0)); fx1 = max(fx0 + 1, min(w_feat, fx1))
            fy0 = max(0, min(h_feat - 1, fy0)); fy1 = max(fy0 + 1, min(h_feat, fy1))

            region = hr_features[:, fy0:fy1, fx0:fx1]  # (C, rh, rw)
            vec = region.reshape(C, -1).mean(dim=1)  # (C,)
            vec = vec / (vec.norm(p=2) + 1e-8)       # L2 normalize
            return vec
     
        with torch.no_grad():
            vecs = torch.stack([box_to_featvec(b) for b in single_instance_boxes])  # (N, C)
        N = vecs.shape[0]
        if N == 1:
            fce = single_instance_boxes[0]
        else:
            c0_idx = 0
            sims0 = (vecs @ vecs[c0_idx])                # cosine sims
            c1_idx = int(torch.argmin(sims0))            # farthest point
            centroids = torch.stack([vecs[c0_idx], vecs[c1_idx]])  
            centroids = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)

            for _ in range(10):  
                sims = vecs @ centroids.T            
                labels = torch.argmax(sims, dim=1)   
                new_centroids = []
                for j in range(2):
                    idx = (labels == j).nonzero(as_tuple=False).squeeze(1)
                    if idx.numel() > 0:
                        mean_j = vecs[idx].mean(dim=0)
                        mean_j = mean_j / (mean_j.norm(p=2) + 1e-8)
                        new_centroids.append(mean_j)
                    else:
                        new_centroids.append(centroids[j])
                new_centroids = torch.stack(new_centroids, dim=0)
                if torch.allclose(new_centroids, centroids, atol=1e-4):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            sizes = torch.tensor([(labels == j).sum().item() for j in range(2)])
            majority = int(torch.argmax(sizes))
            maj_idx = (labels == majority).nonzero(as_tuple=False).squeeze(1)
            sims_to_centroid = (vecs[maj_idx] @ centroids[majority])  
            best_rel = int(torch.argmax(sims_to_centroid))
            best_idx = int(maj_idx[best_rel])
            fce = single_instance_boxes[best_idx]

        boxes = [dae, dge, fce]
        ori_boxes = [[int(b[1]), int(b[0]), int(b[3]), int(b[2])] for b in map(np.array, boxes)]
        query, supports = preprocess(src_img, ori_boxes,device)
      
        if args.fsoc == 'countr':
            samples, boxes,ori_boxes, pos, W, H, new_W, new_H = load_image(test_image_path, ori_boxes)
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, elapsed_time, density_pred, _ = run_one_image(samples, boxes, pos, counter_model, W, H, filename, new_W, new_H)
            vis_output, count, density_pred = vis_countr(src_img, ori_boxes, result, filename, density_pred)
        elif args.fsoc == 'DSALVANet':
            output = counter_model(query,supports)
            vis_output, count, density_pred,_ = vis_DSALVANet(src_img,ori_boxes,output, filename)


        if args.visualize == True:
            os.makedirs(output_path, exist_ok=True)
            counter_output_path = os.path.join(output_path, f'{test_idx}.png')
            cv2.imwrite(counter_output_path, vis_output)
            # set_path = './outputs/' + 'countzes_persenseD'
            # if not os.path.exists(set_path):
            #     os.mkdir('./outputs/countzes_persenseD')
            # counter_output_path = os.path.join(set_path, f'refined_{filename}')
            # cv2.imwrite(counter_output_path,vis_output)

        del hr_features, lr_features, full_feat, hr_image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        print (f'GT_Count = {gt_count}')
        print (f'Pred_Count = {count} \n')
        error = torch.abs(torch.tensor(count) - gt_count).item()
        count_item += 1
        mae += error
        mse += error ** 2

        overall_mae = mae / count_item
        overall_mse = mse / count_item
        overall_mse = overall_mse ** 0.5
        
        print('MAE_Original %.2f, RMSE_Original %.2f \n'%(overall_mae, overall_mse))

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def get_class_name(filename, image_class_path):
    # Open and read the file
    with open(image_class_path, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Split the line by tab to separate filename and class name
            file, class_name = line.strip().split('\t')
            # Check if the filename matches the given filename
            if file == filename:
                return class_name
    return None

def _peaks_from_sim(sim_clip, bbox_xyxy, num_points=16, min_distance=8,
                    start_percentile=80, min_percentile=50, step=10):

    x0, y0, x1, y1 = map(int, bbox_xyxy)
    H, W = sim_clip.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), dtype=int)

    crop = sim_clip[y0:y1+1, x0:x1+1]
    crop_blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0, sigmaY=1.0)

    percentile = start_percentile
    pts = []
    while percentile >= min_percentile and len(pts) < num_points:
        thr = np.percentile(crop_blur, percentile)
        peaks = peak_local_max(
            crop_blur,
            min_distance=min_distance,
            threshold_abs=thr,
            exclude_border=False
        )
        pts = [[x0 + int(px), y0 + int(py)] for (py, px) in peaks]
        if len(pts) >= num_points:
            break
        percentile -= step

    if len(pts) > num_points:
        pts = sorted(pts, key=lambda p: sim_clip[p[1], p[0]], reverse=True)[:num_points]

    return np.array(pts, dtype=int)


def compute_clip_entropy_masked(sim_map, mask, bins=20):
    m = (mask > 0)
    if not np.any(m):
        return np.inf
    vals = sim_map[m].ravel()
    if np.all(vals == vals[0]):
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return np.inf
    hist = (hist + 1e-8) / (hist.sum() + 1e-8 * len(hist))
    return entropy(hist)  # natural log


def SSES(              # Similarity guided SAM-based Exemplar Selection
    test_image,
    bbox_coord,
    predictor,
    clip_model,       
    clip_processor,   
    class_text,      
    device,
    sim_clip=None,
    num_points=16,
    min_distance=5,
    start_percentile=80,
    min_percentile=50,
    step=10,
    filename=None,    # for saving the visualization
):
    
    if sim_clip is None:
        raise ValueError("sim_clip must be provided.")
    
    sim_clip = 1 - sim_clip

    H, W = test_image.shape[:2]
    x0, y0, x1, y1 = map(int, bbox_coord)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return bbox_coord, None, None
    

    global_vals = sim_clip.ravel()
    global_sorted = np.sort(global_vals)
    N_global = global_sorted.size
    points = _peaks_from_sim(
        sim_clip, [x0, y0, x1, y1],
        num_points=num_points,
        min_distance=min_distance,
        start_percentile=start_percentile,
        min_percentile=min_percentile,
        step=step
    )

    if len(points) == 0:
        return bbox_coord, None, None
    

    predictor.set_image(test_image)
    candidates = []
    bins = 20                      
    Hmax = np.log(bins)            
    w_sim = 0.5                    
    w_ent = 0.5                    

    for pt in points:
        masks, _, _, _ = predictor.predict(
            point_coords=pt[None, :],
            point_labels=np.array([1]),
            multimask_output=False
        )
        mask = masks[0]

        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if bx0 < x0 or bx1 > x1 or by0 < y0 or by1 > y1:
            continue
        fg = sim_clip[mask > 0]
        ent = compute_clip_entropy_masked(sim_clip, mask, bins=bins)
        if not np.isfinite(ent):
            continue
        ent_norm = float(np.clip(ent / (Hmax + 1e-12), 0.0, 1.0)) 

        idxs = np.searchsorted(global_sorted, fg, side='right')
        pct_ranks = idxs / N_global  # in [0,1]

        sim_mean_pct = float(pct_ranks.mean())
        sim_score = sim_mean_pct

        comp = w_sim * sim_score + w_ent * (1.0 - ent_norm)

        candidates.append({
            "score": comp,
            "mask": mask,
            "bbox": [bx0, by0, bx1, by1],
            "ent": float(ent_norm)
        })

    coarse_mask = np.zeros(sim_clip.shape, dtype=np.uint8)
    coarse_mask[y0:y1+1, x0:x1+1] = 1

    fg = sim_clip[coarse_mask > 0]
    if fg.size > 0:
        coarse_ent = compute_clip_entropy_masked(sim_clip, coarse_mask, bins=bins)
        if np.isfinite(coarse_ent):
            ent_norm = float(np.clip(coarse_ent / (Hmax + 1e-12), 0.0, 1.0))

            idxs      = np.searchsorted(global_sorted, fg, side='right')
            pct_ranks = idxs / N_global
            mean_pct  = float(pct_ranks.mean())
            sim_score = mean_pct

            comp = w_sim * sim_score + w_ent * (1.0 - ent_norm)

            candidates.append({
                "score": comp,
                "mask":  coarse_mask.astype(bool),
                "bbox":  [x0, y0, x1, y1],
                "ent":   float(ent_norm)
            })

    if not candidates:
        return bbox_coord, None, None   
    best = max(candidates, key=lambda c: c["score"])
    return best["bbox"], best["mask"], best["ent"]


def cluster_counts(sam_box_count, box_list, filename, save_path='./outputs/cluster_count'):
    if not sam_box_count:
        raise ValueError("Empty sam_box_count list.")

    counts_array = np.array(sam_box_count)
    std_dev = np.std(counts_array)  
    
    kde_pseudo_gt = None
    xs, kde_vals = None, None
    try:
        if len(counts_array) > 1 and std_dev > 1e-6:
            kde = gaussian_kde(counts_array)
            xs = np.linspace(min(counts_array), max(counts_array), 500)
            kde_vals = kde(xs)
            kde_pseudo_gt = int(xs[np.argmax(kde_vals)])
        else:
            raise ValueError("Too few unique values for KDE")
    except Exception as e:
        print(f"[Warning] KDE failed for {filename} ({str(e)}). Falling back to mean.")
        if len(set(counts_array)) == 1:
            kde_pseudo_gt = int(counts_array[0]) 
        else:
            kde_pseudo_gt = int(np.round(np.mean(counts_array))) 

    lower_bound = kde_pseudo_gt - 1 * std_dev
    upper_bound = kde_pseudo_gt + 1 * std_dev
    kde_cluster_boxes = [i for i, c in enumerate(sam_box_count) if lower_bound <= c <= upper_bound]

    return {
        'kde_pseudo_gt': kde_pseudo_gt,
        'kde_cluster_boxes': kde_cluster_boxes
    }


def normalize_sim_map(sim_map):
    sim_min = sim_map.min()
    sim_max = sim_map.max()
    norm_map = (sim_map - sim_min) / (sim_max - sim_min + 1e-8)
    return norm_map

def normalize(arr):
    arr = np.array(arr)
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)
    return (arr - min_val) / (max_val - min_val + 1e-8)

def process_box(args: Tuple):
    """
    Args:
        args: tuple containing (sam_box, img_path, counter_model, args, src_img, filename, device)
    """
    sam_box, img_path, counter_model, args, src_img, filename, device = args
    x0, y0, x1, y1 = map(int, sam_box)
    sam_box = [[y0, x0, y1, x1]]

    try:
        if args.fsoc == 'countr':
            samples, boxes, sam_box_used, pos, W, H, new_W, new_H = load_image(img_path, sam_box)
            samples = samples.unsqueeze(0).to(device, non_blocking=True)
            boxes = boxes.unsqueeze(0).to(device, non_blocking=True)
            result, elapsed_time, density_pred, _ = run_one_image(samples, boxes, pos, counter_model, W, H, filename, new_W, new_H)
            _, count,_ = vis_countr(src_img, sam_box_used, result, filename, density_pred)

        elif args.fsoc == 'DSALVANet':
            query, supports = preprocess(src_img, sam_box, device)
            output = counter_model(query, supports)
            _, count, _,_ = vis_DSALVANet(src_img, sam_box, output, filename)

        return int(count)

    except Exception as e:
        print(f"Error processing box {sam_box}: {e}")
        return -1  # fallback value


def compute_clip_entropy(clip_sim_map, box, bins=20):
    x0, y0, x1, y1 = map(int, box)
    if y1 <= y0 or x1 <= x0:
        return np.inf
    region = clip_sim_map[y0:y1, x0:x1]
    if region.size == 0:
        return np.inf   
    flat_region = region.flatten()
    if np.all(flat_region == flat_region[0]):
        return 0.0
    hist, _ = np.histogram(flat_region, bins=bins, range=(0, 1), density=True)
    hist += 1e-8  
    hist /= hist.sum()  
    return entropy(hist)


def get_masks_and_boxes(predictor, image, point_prompts):
    masks_list = []
    boxes_list = []
    predictor.set_image(image)

    for pt in point_prompts:
        input_point = np.array([pt]) 
        input_label = np.array([1])       

        masks, scores, logits, low_res_logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  
        )

        mask = masks[0] 
        masks_list.append(mask)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x0, y0 = np.min(x_indices), np.min(y_indices)
            x1, y1 = np.max(x_indices), np.max(y_indices)
            boxes_list.append([x0, y0, x1, y1])
        else:
            boxes_list.append([0, 0, 0, 0])
    return masks_list, boxes_list

def compute_clip_similarity_map(img_rgb, class_name, clip_model, clip_processor, device):
    inputs = clip_processor(text=[class_name], images=[img_rgb], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, C)
        text_outputs = clip_model.get_text_features(input_ids=inputs["input_ids"],
                                                    attention_mask=inputs["attention_mask"])  # (1, C)
    patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, 768)
    patch_embeds = clip_model.visual_projection(patch_embeds)  # (1, num_patches, 512)
    patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)
    text_emb = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
    sim = (patch_embeds @ text_emb.unsqueeze(-1)).squeeze(-1)  # (1, num_patches)
    num_patches = sim.shape[-1]
    side = int(math.sqrt(num_patches))
    sim_map = sim.reshape(1, 1, side, side)  # (1,1,side,side)
    sim_map = F.interpolate(sim_map,
                            size=img_rgb.shape[:2],
                            mode="bilinear",
                            align_corners=False).squeeze(0).squeeze(0)  # (H, W)
    return sim_map.detach().cpu().numpy()


def get_text_description(filename, json_data):
    if filename in json_data:
        return json_data[filename].get("text_description", "No description found.")
    else:
        return "Filename not found in the dataset."
    

def filter_peak_prompts(peak_points, detections):
    filtered_points = []
    boxes = np.array(detections.xyxy)  # shape: (M, 4) where each is [x0, y0, x1, y1]
    for y, x in peak_points:
        inside_any_box = False
        for x0, y0, x1, y1 in boxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                inside_any_box = True
                break
        if inside_any_box:
            filtered_points.append([x, y])  # SAM expects (x, y)
    return filtered_points

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    main()
