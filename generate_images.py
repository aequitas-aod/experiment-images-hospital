from pathlib import Path
from typing import List
import numpy as np
import re
from PIL import Image
import os
from skimage import exposure, measure
import scipy
import scipy.ndimage

def generate_path_pairs(starting_path: Path) -> List[Path]:
    # Ps. I could shorten this by only using globs, but I trust regexps more
    img_mask_pairs = []
    # Find all folders named "personaX" in disease folder
    persona_regexp = r"^persona\d+$"
    folders = [x for x in starting_path.glob('*') if x.is_dir()]
    folders = [x for x in folders if re.search(persona_regexp, x.name) is not None]
    # Find all folders named "exampleX" in each persona folder
    example_regexp = r"^example\d+$"
    example_folders = []
    for f in folders:
        example_folders += f.glob('*')
    example_folders = [x for x in example_folders if 
                       x.is_dir() and re.search(example_regexp, x.name) is not None]
    for ex_folder in example_folders:
        # Find the cropped images
        ex_name = ex_folder.name
        crop_regexp = rf"^{ex_name}_\d+.png$" 
        crop_images = [x for x in ex_folder.glob('*') if re.search(crop_regexp, x.name) is not None]

        # Find the masks
        for img in crop_images:
            mask = Path(str(img).replace(".png", "_mask.png"))
            if mask.exists():
                img_mask_pairs.append((img, mask))
    return img_mask_pairs

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes

def generate_croppings(disease_folder, img_mask_pairs : List[Path], crop_size: List[int], crop_shift:int,
                       min_mask_ratio: float, nms_threshold: float = 0.4) -> List[np.ndarray]:
    
    out_folder=os.path.join(disease_folder, "crops")
    os.makedirs(out_folder, exist_ok=True)
    low_contrast_folder = os.path.join(disease_folder, "low_contrast")
    os.makedirs(low_contrast_folder, exist_ok=True)
    c = 1
    n_low_contrast = 0
    n_blurry = 0
    i = 1
    for img_path, mask_path in img_mask_pairs:

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        # 0-1 encode the mask
        np_mask = np.array(mask)/255
        if len(np_mask.shape) == 3:
            np_mask = np_mask[:,:,0]
        mask = Image.fromarray(np_mask > 0.3)


        # Let's just work on the mask
        w, h = mask.size
        # Skip images where the crop does not fit at all
        if w < crop_size[0] or h < crop_size[1]:
            continue

        # Find all cropping rectangles 
        # Define the maximum x and y coordinates of the top-left corner of the crop rectangle
        max_x = w - crop_size[0]
        max_y = h - crop_size[1]

        x_starts = np.arange(0, max_x, crop_shift)
        y_starts = np.arange(0, max_y, crop_shift)

        valid_crops = []
        scores = []
        for x in x_starts:
            for y in y_starts:
                # Obtain each possible cropping and then evaluate its positive mask coverage 
                # (ie. the percentage of positive pixels in the mask)
                cropped = mask.crop((x, y, x+crop_size[0], y+crop_size[1]))
                coverage = np.mean(np.array(cropped))

                # Discard croppings with not enough positives
                if coverage > min_mask_ratio:
                    valid_crops.append((x,y,x+crop_size[0],y+crop_size[1]))
                    scores.append(coverage)
        if not valid_crops:
            continue
        # Perform nms to remove some of the overlapping croppings, preferring those with higher coverage
        nms_boxes = nms(np.asarray(valid_crops), np.asarray(scores), nms_threshold)

        for box in nms_boxes:
            cropped_image = img.crop(box)
            cropped_mask = mask.crop(box)

            # Create a colored mask
            np_image = np.array(cropped_image)[:,:,:3]
            np_mask = np.array(cropped_mask)

            masked = np_image * np.repeat(np.expand_dims(np_mask,-1),3, axis=-1)
            unmasked = np_image * np.repeat(np.expand_dims(1-np_mask,-1),3, axis=-1)
            def get_dom_color(ar):
                import scipy

                shape = ar.shape
                ar = ar.reshape(np.prod(shape[:2]), shape[2]).astype(float)

                codes, dist = scipy.cluster.vq.kmeans(ar, 5)
                codes = [c for c in codes if np.mean(c) != 0]
                if len(codes) == 0:
                    codes = [(0,0,0)]

                vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
                counts, bins = np.histogram(vecs, len(codes))    # count occurrences

                index_max = np.argmax(counts)                    # find most frequent
                peak = codes[index_max]
            
                return peak

            avg_disease_color = get_dom_color(masked)
            avg_back_color = get_dom_color(unmasked)

            cool_img = np.zeros((256,256,3))
            cool_img[np_mask,:] = avg_disease_color
            cool_img[~np_mask,:] = avg_back_color

            cool_img = Image.fromarray(np.uint8(cool_img))
            if exposure.is_low_contrast(cropped_image):
                n_low_contrast += 1
                cropped_mask.save(Path(low_contrast_folder)/f"{c:04d}_mask.png")
                cropped_image.save(Path(low_contrast_folder)/f"{c:04d}.png")
                cool_img.save(Path(low_contrast_folder)/f"{c:04d}_mask2.png")
                continue
            
            cropped_mask.save(Path(out_folder)/f"{c:04d}_mask.png")
            cropped_image.save(Path(out_folder)/f"{c:04d}.png")
            cool_img.save(Path(out_folder)/f"{c:04d}_mask2.png")

            c += 1
        
        i += 1
    print(f"Generated {n_blurry+n_low_contrast+c} images")
    print(f"Skipped {n_low_contrast} low contrast images")