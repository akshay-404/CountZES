import cv2
import numpy as np
import torch
import matplotlib.pyplot  as plt
import math
import os

def apply_scoremap(image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=np.float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def resize_density_sum_preserving(density_hw, out_w, out_h):
    """density_hw: [H, W] float. Returns [out_h, out_w] with same total sum."""
    H, W = density_hw.shape
    resized = cv2.resize(density_hw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    scale = (W * H) / float(out_w * out_h)  # multiply so global sum is preserved
    return resized * scale

def vis_DSALVANet(src_img, boxes, model_output, test_idx):
        sum = torch.sum(model_output)
        pre_cnt = int(math.ceil(sum))
        # pre_cnt = int(torch.sum(model_output))
        h_orig, w_orig = src_img.shape[0], src_img.shape[1]
        density_pred = model_output.squeeze(0)
        density_pred = density_pred.permute(1, 2, 0)  # c x h x w -> h x w x c
        density_pred = density_pred.cpu().detach().numpy()
        density_pred = cv2.resize(density_pred, (w_orig, h_orig))
        density_pred = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
        density_pred= np.stack((density_pred,) * 3, axis=-1)


        density_raw = model_output.squeeze(0)  # [C,H,W] or [1,H,W]
        if density_raw.ndim == 3 and density_raw.shape[0] == 1:
                density_raw = density_raw[0]          # -> [H,W]
        density_raw = density_raw.detach().cpu().numpy().astype('float32')

        H0, W0 = density_raw.shape
        h_orig, w_orig = src_img.shape[:2]

        # IMPORTANT: preserve the integral when resizing to image size
        density_raw = resize_density_sum_preserving(density_raw, w_orig, h_orig)

        # Sanity check: total count should match the model’s global sum (within tolerance)
        global_count_model = float(model_output.sum().item())
        global_count_resized = float(density_raw.sum())
        if not np.isfinite(global_count_resized) or abs(global_count_model - global_count_resized) > 0.5:
                print(f"[Warn] density sum mismatch: model={global_count_model:.2f}, resized={global_count_resized:.2f}")

        gray = cv2.cvtColor((density_pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        output = apply_scoremap(src_img, density_pred)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        for box in boxes:
            y1, x1, y2, x2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.putText(output, "Result:{0}".format(pre_cnt), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
        return output , pre_cnt, gray, density_raw