import re
import numpy as np
from starvector2.utils.svg_util import get_svg_original_size
import os

def reward_svg_edit_distance(svg_gt: str, svg_pred: str) -> float:
    """
    Compute the edit distance penalty for the SVG string.
    """
    if not svg_gt or not svg_pred:
        return 0.0
    edit_distance = edit_distance(svg_gt, svg_pred) / max(len(svg_gt), len(svg_pred))
    # Map normalized edit distance [0,1] to score in [-1,1], higher is better
    score = np.clip(1.0 - 2.0 * edit_distance, -1.0, 1.0)
    return 1 - edit_distance

# def reward_svg_

def reward_svg_length_quadratic(svg_gt: str, svg_pred: str, svg_gt_len_override = None) -> float:
     """
     Compute the length penalty for the SVG string.
     1 - (max(0,(x-y/2))/y)**2, clip to -1,1
     
     half: 1
     same:  0.75
     1.5x: 0
     2x: -1.25 (-1)
     """
     # Calculate the length of the SVG string
     svg_pred_len = len(svg_pred)
     
     # If svg_gt_len_override is provided, use it; otherwise, calculate the length of the ground truth SVG
     if svg_gt_len_override:
         svg_gt_len = svg_gt_len_override
     else:
         svg_gt_len = len(svg_gt)
     
     return np.clip(1 - (max(0,(svg_pred_len-svg_gt_len/2))/svg_gt_len)**2, -1,1)


def reward_svg_length_juan(svg_gt: str, svg_pred: str, svg_gt_len_override = None) -> float:
    """
    Compute the length penalty for the SVG string.
    1 - (max(0,(x-y/2))/y)**2, clip to -1,1
    
    half: 1
    same:  0.75
    1.5x: 0
    2x: -1.25 (-1)
    """
    try:
        # Calculate the length of the SVG string
        svg_pred_len = len(svg_pred)
            
        # determine ground-truth length
        if svg_gt_len_override is not None:
            svg_gt_len = svg_gt_len_override
        else:
            svg_gt_len = len(svg_gt)

        # avoid division-by-zero
        if svg_gt_len <= 0:
            return 0.0

        # compute prediction-to-gt length ratio
        ratio = svg_pred_len / svg_gt_len

        # strongly penalize if too short (< 50% of GT)
        if ratio < 0.5:
            return -1.0

        # quadratic penalty around perfect match (ratio=1)
        reward = 1 - (ratio - 1) ** 2
        return np.clip(reward, -1, 1)
    except Exception as e:
        print(f"Error in reward_svg_length: {e}")
        return 0.0

def reward_aspect_ratio(svg_gt: str, svg_pred: str) -> float:
    """
    Compute the aspect ratio penalty for the SVG string.
    """
    try:
        svg_gt_size = get_svg_original_size(svg_gt)
        svg_pred_size = get_svg_original_size(svg_pred)
        # validate returned sizes and prevent division errors
        if svg_gt_size is None or svg_pred_size is None:
            return -1.0
        gt_w, gt_h = svg_gt_size
        pred_w, pred_h = svg_pred_size
        if gt_w is None or gt_h is None or pred_w is None or pred_h is None or gt_h == 0 or pred_h == 0:
            return -1.0
        original_aspect_ratio = gt_w / gt_h
        predicted_aspect_ratio = pred_w / pred_h
        aspect_ratio_accuracy = min(original_aspect_ratio, predicted_aspect_ratio) / max(original_aspect_ratio, predicted_aspect_ratio)

        return aspect_ratio_accuracy
    except Exception as e:
        print(f"Error in reward_aspect_ratio: {e}")
        return -1.0

## -- old code --

def reward_svg_special_token(predict_str: str) -> float:
    """
    Check if the SVG string is well-formed.
    """
    pattern = re.compile(r"<svg\b[^>]*>.*</svg>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else -1.0


def reward_svg_syntax(clean, fixable) -> float:
    """
    Check if the SVG have syntax errors, and if it is fixable
    """
    if clean:
        return 1.0
    if fixable:
        return 0.0
    return -1.0

def reward_renderability(renderable) -> float:
    """
    Check if the SVG is renderable
    """
    if renderable:
        return 1.0
    return -1.0

