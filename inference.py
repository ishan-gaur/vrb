import cv2
from sklearn.mixture import GaussianMixture


import argparse
import os
import random
import numpy as np
import torch
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from  PIL  import  Image
from lang_sam import LangSAM
transform  = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomGrayscale(p=0.05),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])





model = LangSAM()
def compute_heatmap(points, image_size, k_ratio=3.0):
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        col = int(x)
        row = int(y)
        try:
            heatmap[col, row] += 1.0
        except:
            col = min(max(col, 0), image_size[0] - 1)
            row = min(max(row, 0), image_size[1] - 1)
            heatmap[col, row] += 1.0
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap.transpose()
    return heatmap

def run_inference(net, image_pil, objects=None, overlap_thresh=None, max_boxes=None, heatmap_cutoff=25, device=None): 
    assert isinstance(image_pil, Image.Image), "image_pil must be a PIL image"
    assert isinstance(objects, list), "objects must be a list of strings"
    assert overlap_thresh is None or 0 < overlap_thresh <= 1, "overlap_thresh must be in (0,1]"
    assert max_boxes is None or max_boxes > 0, "max_boxes must be greater than 0"
    assert 0 <= heatmap_cutoff <= 255, "heatmap_cutoff must be in [0,255]"

    if objects is None:
        objects = ['cup', 'drawer', 'potlid', 'microwave']

    masks_list, bboxes, phrases_list, logits_list = [], [], [], []
    for obj in objects: 
        with torch.no_grad(): 
            masks, boxes, phrases, logits = model.predict(image_pil, obj)
            # select the one with the highest logit
            if len(logits) == 0:
                continue
            i = torch.argmax(logits)
            mask, box, phrase, logit = masks[i], boxes[i], phrases[i], logits[i]
        masks_list.append(mask)
        bboxes.append(box)
        phrases_list.append(phrase)
        logits_list.append(logit)

    assert len(bboxes) != 0, "No objects found in the image"

    # go through the boxes, if any of the masks overlap by more than 0.5 of the smaller, remove the one with the lower logit
    if not overlap_thresh is None:
        i, j = 0, 1
        while i < len(masks_list) - 1:
            if j > len(masks_list) - 1:
                i += 1
                j = i + 1
                continue
            threshold = overlap_thresh * min(torch.sum(masks_list[i]), torch.sum(masks_list[j]))
            if torch.sum(masks_list[i] * masks_list[j]) > threshold:
                if logits_list[i] > logits_list[j]:
                    masks_list.pop(j)
                    bboxes.pop(j)
                    phrases_list.pop(j)
                    logits_list.pop(j)
                else:
                    masks_list.pop(i)
                    bboxes.pop(i)
                    phrases_list.pop(i)
                    logits_list.pop(i)
            j += 1

        assert len(bboxes) != 0, "Filtering for overlap removed all objects"
    
    if not max_boxes is None:
        if len(bboxes) > max_boxes:
            logits_list = torch.stack(logits_list)
            _, idxs = torch.topk(logits_list, k=max_boxes, dim=0)
            masks_list = [masks_list[i] for i in idxs]
            bboxes = [bboxes[i] for i in idxs]
            phrases_list = [phrases_list[i] for i in idxs]
            logits_list = [logits_list[i] for i in idxs]
        
    print("Bounding boxes max, actual", max_boxes, len(bboxes))

    contact_points = []
    trajectories = []
    for box in bboxes: 
        y1, x1, y2, x2 = box
        bbox_offset = 20
        y1, x1, y2, x2 = int(y1) - bbox_offset, int(x1) - bbox_offset , int(y2) + bbox_offset, int(x2) + bbox_offset

        width = y2 - y1
        height = x2 - x1
        
        diff = width - height
        if width > height:
            y1 += int(diff / np.random.uniform(1.5, 2.5))
            y2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))
        else:
            diff = height - width
            x1 += int(diff / np.random.uniform(1.5, 2.5))
            x2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))

        img = np.asarray(image_pil)
        input_img = img[x1:x2, y1:y2]
        inp_img = Image.fromarray(input_img)
        inp_img = transform(inp_img).unsqueeze(0)
        gm = GaussianMixture(n_components=3, covariance_type='diag')
        centers = []
        trajs = []
        traj_scale = 0.1
        with torch.no_grad(): 
            if device is not None:
                inp_img = inp_img.to(device)
            ic, pc = net.inference(inp_img, None, None)
            pc = pc.cpu().numpy()
            ic = ic.cpu().numpy()
            i = 0
            w, h = input_img.shape[:2]
            sm = pc[i, 0]*np.array([h, w])
            centers.append(sm)
            trajs.append(ic[0, 2:])
        gm.fit(np.vstack(centers))
        cp, indx = gm.sample(50)
        x2, y2 = np.vstack(trajs)[np.random.choice(len(trajs))]
        dx, dy = np.array([x2, y2])*np.array([h, w]) + np.random.randn(2)*traj_scale
        scale = 40/max(abs(dx), abs(dy))
        adjusted_cp = np.array([y1, x1]) + cp
        contact_points.append(adjusted_cp)
        trajectories.append([x2, y2, dx, dy])
    

    original_img = np.asarray(image_pil)
    # k_ratio sets the size of the gaussian kernel used to blur the heatmap
    hmap = compute_heatmap(np.vstack(contact_points), (original_img.shape[1],original_img.shape[0]), k_ratio = 6)
    hmap = (hmap * 255).astype(np.uint8)
    hmap_mask = np.where(hmap > heatmap_cutoff, 1, 0).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, colormap=cv2.COLORMAP_JET)
    overlay = (0.6 * original_img +  0.4 * hmap).astype(np.uint8) * hmap_mask[:, :, None]
    overlay += (original_img * (1 - hmap_mask[:, :, None])).astype(np.uint8)

    plt.clf()
    plt.imshow(overlay)

    print("# contact points: ", len(contact_points))
    for i, cp in enumerate(contact_points):
        x2, y2, dx, dy = trajectories[i]
        scale = 60/max(abs(dx), abs(dy))
        x, y = cp[:, 0] , cp[:, 1]
        plt.arrow(int(np.mean(x)), int(np.mean(y)), scale*dx, -scale*dy, color='white', linewidth=2.5, head_width=12)


    plt.axis('off')
    img_buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    im = Image.open(img_buf)
    plt.close()
    return im
