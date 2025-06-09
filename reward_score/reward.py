import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch
import math
from svgpathtools import svgstr2paths
# from reward_edge import cannyboximage
import numpy as np
# import cv2

def get_reward_function(reward_config: dict):
    """
    Given a reward config dictionary (from YAML), return a list of reward functions with their weights.
    
    Example reward_config from YAML:
    rewards:
      reward_L2_reconstruction:
        weight: 1.0
      reward_token_len_deviation:
        weight: 1.0

    """
    print(f"reward_config: {reward_config}")
    reward_functions = []
    
    # Map reward names to their function implementations
    reward_map = {
        'reward_token_len_deviation': reward_token_len_deviation,
        'reward_compilation': None,  # TODO: Implement if needed
        'reward_dinoscore': reward_dinoscore,
        'reward_L2_reconstruction': reward_L2_reconstruction,
        'reward_SSIM': reward_SSIM,
        'reward_SVG_format': reward_SVG_format,
        'reward_color_histogram': reward_color_histogram,
    }
    
    for reward_name, config in reward_config.items():
        if reward_name not in reward_map:
            print(f"Warning: Unknown reward function {reward_name}")
            continue
            
        if reward_map[reward_name] is None:
            print(f"Warning: Reward function {reward_name} is not implemented yet")
            continue
            
        weight = config.get('weight', 1.0)
        reward_func = reward_map[reward_name]
        
        # Extract all other config parameters to pass to the reward function
        reward_kwargs = {k: v for k, v in config.items() if k != 'weight'}
        
        # Create a wrapped function that applies the weight and passes other parameters
        def weighted_reward(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, 
                          func=reward_func, w=weight, func_kwargs=reward_kwargs, **kwargs):
            # Combine the config kwargs with any runtime kwargs
            combined_kwargs = {**func_kwargs, **kwargs}
            rewards = func(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, 
                         weight=w, **combined_kwargs)
            return [r * w for r in rewards]
        
        weighted_reward.__name__ = f"{reward_name}"
        reward_functions.append(weighted_reward)
    
    if not reward_functions:
        raise ValueError("No valid reward functions found in configuration")
        
    return reward_functions

def l2_distance(image1, image2, masked_l2=False, use_edge=False, edge_thresh=(50,150)):
    """
    Computes the L2 (mean squared error) distance between two images.
    
    Args:
        image1: First image (numpy array or PIL Image)
        image2: Second image (numpy array or PIL Image)
        masked_l2: If True, only compute MSE over non-white pixels
        use_edge: If True, compute edges and corresponding masks bsed on edges
        edge_thresh: Only used if use_edge is true. High and low threshold for hysteresis
    
    Returns:
        float: The mean squared error between the two images, or float('inf') if all pixels are masked out
    """
    image1_tensor = ToTensor()(image1)
    image2_tensor = ToTensor()(image2)
    
    assert not(use_edge and not masked_l2), "use_edge requires masked_l2 to be True"
    assert (use_edge and (edge_thresh is not None))^(not use_edge), "Use edge thresh if using use edge"

    if masked_l2:

        if not use_edge:

            # Create masks for non-white pixels in each image
            mask1 = (image1_tensor != 1).any(dim=0)
            mask2 = (image2_tensor != 1).any(dim=0)
        # else:
        #     mask1=cannyboximage(image1,low_thresh=edge_thresh[0],high_thresh=edge_thresh[1])
        #     mask2=cannyboximage(image2,low_thresh=edge_thresh[0],high_thresh=edge_thresh[1])
        #     mask1=ToTensor()(mask1)
        #     mask2=ToTensor()(mask2)
        # Use union of masks (any pixel that is non-white in either image)
        common_mask = (mask1 | mask2).float()
        
        # Apply mask to squared differences
        squared_diff = ((image1_tensor - image2_tensor) ** 2) * common_mask.unsqueeze(0)
        
        # Calculate masked MSE
        num_pixels = common_mask.sum()
        if num_pixels > 0:  # Avoid division by zero
            mse = squared_diff.sum() / num_pixels
            return mse.item()
        else:
            # If no pixels in mask, return infinity
            return float('inf')
    else:
        # Standard MSE calculation
        mse = F.mse_loss(image1_tensor, image2_tensor)
        return mse.item()
    
def reward_token_len_deviation(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed,
                               deviation_threshold=50, penalty_weight=0.01, min_reward=-1.0, 
                               long_svg_penalty_multiplier=2.0, **kwargs):
    """
    Computes a reward based on how close the predicted SVG token length is to the ground truth.
    Favors shorter SVGs over longer ones when both are outside the acceptable deviation threshold.

    Parameters:
        input_images: list of input images (unused in this function but included for API consistency)
        svg_rasters: list of rasterized SVG images (unused here)
        gt_svgs: list of ground truth SVG strings
        predicted_svg_raws: list of predicted SVG strings (raw tokens)
        predicted_svg_processed: processed SVG predictions (unused here)
        deviation_threshold: int, allowed offset in token length without penalty.
        penalty_weight: float, scaling factor for penalizing tokens beyond the threshold.
        min_reward: float, lower bound for the reward (default: -1.0) to prevent extreme negative values.
        long_svg_penalty_multiplier: float, multiplier for the penalty when SVG is longer than ground truth.
        **kwargs: Additional keyword arguments

    Returns:
        rewards: list of rewards, one per SVG, with a maximal reward (1.0) when within the threshold,
                 decreasing linearly for deviations beyond it but not dropping below min_reward.
                 Penalties are harsher for SVGs that are longer than the ground truth.
    """
    rewards = []
    for gt_svg, predicted_svg in zip(gt_svgs, predicted_svg_raws):
        gt_len = len(gt_svg)
        pred_len = len(predicted_svg)
        signed_diff = pred_len - gt_len  # Positive when prediction is longer
        abs_diff = abs(signed_diff)
        
        if abs_diff <= deviation_threshold:
            reward = 1.0  # Perfect reward if within the acceptable deviation
        else:
            # Apply different penalties based on whether SVG is too short or too long
            if signed_diff > 0:  # Predicted SVG is longer than ground truth
                # Apply stronger penalty for longer SVGs
                effective_penalty = penalty_weight * long_svg_penalty_multiplier
            else:  # Predicted SVG is shorter than ground truth
                effective_penalty = penalty_weight
                
            # Penalize the extra deviation beyond the threshold, but clamp the minimum value
            reward = max(1.0 - effective_penalty * (abs_diff - deviation_threshold), min_reward)
        
        rewards.append(reward)
    
    return rewards

def reward_dinoscore(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, device='cuda', white_svg_penalty=-1.0, **kwargs):
    """
    Computes a reward based on the DINO similarity score between input images and their SVG reconstructions.
    DINO scores capture semantic similarity rather than just pixel-level differences.
    
    Args:
        input_images: List of input images (numpy arrays or PIL Images)
        svg_rasters: List of rasterized SVG images
        gt_svgs: List of ground truth SVG strings (unused)
        predicted_svg_raws: List of predicted SVG strings (unused)
        predicted_svg_processed: List of processed SVG predictions (unused)
        device: Device to run the DINO model on (default: 'cuda')
        white_svg_penalty: Fixed penalty for all-white SVGs (default: -1.0)
        **kwargs: Additional keyword arguments
    
    Returns:
        list: DINO similarity scores for each image pair, or penalty for mostly white images
    """
    # Import here to avoid circular imports
    from starvector.metrics.compute_dino_score import DINOScoreCalculator
    
    # Initialize the DINO score calculator (lazy loading)
    if not hasattr(reward_dinoscore, 'dino_calculator'):
        reward_dinoscore.dino_calculator = DINOScoreCalculator(device=device)
    
    rewards = []
    raw_dino_scores = []
    
    # Process each image pair
    for input_image, svg_raster in zip(input_images, svg_rasters):
        # Convert SVG raster to tensor to check if it's mostly white
        image_tensor = ToTensor()(svg_raster)
        
        # Check if the SVG is all white (or nearly all white)
        mask = (image_tensor != 1).any(dim=0)
        non_white_pixels = mask.float().sum().item()
        total_pixels = mask.numel()
        
        # If less than 10% of pixels are non-white, consider it an all-white SVG
        if non_white_pixels < total_pixels * 0.1:
            rewards.append(white_svg_penalty)
            raw_dino_scores.append(0.0)
            continue
        
        # Calculate DINO similarity score (0-1 range)
        dino_score = reward_dinoscore.dino_calculator.calculate_DINOv2_similarity_score(
            gt_im=input_image, 
            gen_im=svg_raster
        )
        raw_dino_scores.append(dino_score)
        
        rewards.append(dino_score)
    
    return rewards

def reward_L2_reconstruction(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, 
                              max_reward=1.0, sensitivity=5.0, masked=True, use_edge=False, white_svg_penalty=-1.0, **kwargs):
    """
    Maps MSE values to rewards using an exponential decay function, with optional masking
    for non-white pixels. Also penalizes all-white SVGs with a fixed penalty.
    
    Args:
        input_images: List of input images (numpy arrays or PIL Images)
        svg_rasters: List of rasterized SVG images
        gt_svgs: List of ground truth SVG strings (unused)
        predicted_svg_raws: List of predicted SVG strings (unused)
        predicted_svg_processed: List of processed SVG predictions (unused)
        max_reward: Maximum reward value (default: 1.0)
        sensitivity: Controls how quickly reward drops as MSE increases (default: 5.0)
                     Higher values = more sensitive to small errors
        masked: Whether to use masking for non-white pixels (default: True)
        white_svg_penalty: Fixed penalty for all-white SVGs (default: -1.0)
        **kwargs: Additional keyword arguments
    
    Returns:
        list: Reward values between 0 and max_reward (or white_svg_penalty for all-white SVGs)
    """
    rewards = []
    raw_mse_values = []
    
    for input_image, svg_raster in zip(input_images, svg_rasters):
        # Convert images to tensors
        image2_tensor = ToTensor()(svg_raster)
        
        # Check if the SVG is all white (or nearly all white)
        mask2 = (image2_tensor != 1).any(dim=0)
        non_white_pixels = mask2.float().sum().item()
        total_pixels = mask2.numel()
        
        # If less than 0.5% of pixels are non-white, consider it an all-white SVG
        if non_white_pixels < total_pixels * 0.005:
            rewards.append(white_svg_penalty)
            raw_mse_values.append(float('inf'))
            continue
        
        # Use the enhanced l2_distance function with masking parameter
        raw_mse = l2_distance(input_image, svg_raster, masked_l2=masked, use_edge=use_edge)
        
        # If masking resulted in no pixels to compare (infinite MSE)
        if raw_mse == float('inf'):
            rewards.append(white_svg_penalty)
            raw_mse_values.append(float('inf'))
            continue
            
        raw_mse_values.append(raw_mse)
        
        # Convert MSE to reward using exponential decay
        reward = max_reward * math.exp(-sensitivity * raw_mse)
        rewards.append(reward)
    
    return rewards

def reward_SVG_compilation(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, **kwargs):
    """
    Computes a reward based on the successful compilation of the predicted SVG strings.

    This function attempts to parse each predicted SVG string using 
    `svgstr2paths`. If any SVG string fails to compile (i.e., an exception is raised 
    during parsing), the function returns a penalized reward of -1.0. Conversely, if all 
    predicted SVG strings compile successfully, it returns a reward of 1.0.
    
    The reward is returned as a single-element list, which maintains consistency with 
    the output of other reward functions.

    Parameters
    ----------
    input_images : list
        List of input images. (This parameter is unused in this function, but is included
        for API consistency with other reward functions.)
    svg_rasters : list
        List of rasterized SVG images. (Unused)
    gt_svgs : list
        List of ground truth SVG strings. (Unused)
    predicted_svg_raws : list
        List of predicted SVG strings to validate for successful compilation.
    predicted_svg_processed : list
        List of processed SVG predictions. (Unused)
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    list
        A single-element list containing:
            -  [1.0] if all predicted SVG strings compile successfully.
            -  [-1.0] if any predicted SVG string fails to compile.
    """
    for predicted_svg in predicted_svg_raws:
        try:
            svgstr2paths(predicted_svg)  # This will raise an exception if the SVG is invalid
        except Exception:  # Catch any exception during SVG parsing
            return [-1.0]
    return [1.0]

def reward_SSIM(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed,
               max_reward=1.0, min_reward=-1.0, white_svg_penalty=-1.0, win_size=11, 
               use_grayscale=False, **kwargs):
    """
    Computes a reward based on the Structural Similarity Index Measure (SSIM) between input images
    and their SVG reconstructions. SSIM better captures perceptual similarity than pixel-wise metrics.
    
    Args:
        input_images: List of input images (numpy arrays or PIL Images)
        svg_rasters: List of rasterized SVG images
        gt_svgs: List of ground truth SVG strings (unused)
        predicted_svg_raws: List of predicted SVG strings (unused)
        predicted_svg_processed: List of processed SVG predictions (unused)
        max_reward: Maximum reward value (default: 1.0)
        min_reward: Minimum reward value for valid comparisons (default: -1.0)
        white_svg_penalty: Fixed penalty for all-white SVGs (default: -1.0)
        win_size: Window size for SSIM calculation (default: 11)
        use_grayscale: If True, convert images to grayscale before computing SSIM (default: False)
        **kwargs: Additional keyword arguments
    
    Returns:
        list: SSIM-based rewards for each image pair, scaled to [min_reward, max_reward]
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("Warning: skimage.metrics not found. Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-image"])
        from skimage.metrics import structural_similarity as ssim
    
    import numpy as np
    rewards = []
    for input_image, svg_raster in zip(input_images, svg_rasters):
        # Convert PIL Images to numpy arrays if necessary
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image)
        if not isinstance(svg_raster, np.ndarray):
            svg_raster = np.array(svg_raster)
        
        # Check if the SVG is all white (or nearly all white)
        if len(svg_raster.shape) == 3:
            # For RGB images, check if all channels are close to white
            non_white_pixels = np.sum(np.any(svg_raster < 250, axis=2))
        else:
            non_white_pixels = np.sum(svg_raster < 250)
            
        total_pixels = svg_raster.shape[0] * svg_raster.shape[1]
        
        # If less than 0.5% of pixels are non-white, consider it an all-white SVG
        if non_white_pixels < total_pixels * 0.005:
            rewards.append(white_svg_penalty)
            continue
        
        # Calculate SSIM
        try:
            if use_grayscale:
                # Convert to grayscale if requested
                if len(input_image.shape) == 3 and input_image.shape[2] == 3:
                    input_image_gray = np.mean(input_image, axis=2).astype(np.uint8)
                else:
                    input_image_gray = input_image
                    
                if len(svg_raster.shape) == 3 and svg_raster.shape[2] == 3:
                    svg_raster_gray = np.mean(svg_raster, axis=2).astype(np.uint8)
                else:
                    svg_raster_gray = svg_raster
                
                ssim_value = ssim(input_image_gray, svg_raster_gray, 
                                  data_range=255, win_size=win_size)
            else:
                # Use multichannel SSIM for RGB images
                multichannel = len(input_image.shape) == 3 and input_image.shape[2] == 3
                if multichannel:
                    ssim_value = ssim(input_image, svg_raster, 
                                     data_range=255, win_size=win_size, 
                                     channel_axis=2)
                else:
                    ssim_value = ssim(input_image, svg_raster, 
                                     data_range=255, win_size=win_size)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            rewards.append(min_reward)
            continue
            
        # Map SSIM value ([-1, 1]) to reward range ([min_reward, max_reward])
        # SSIM values are typically in [0, 1] range for realistic images
        reward = min_reward + (max_reward - min_reward) * (ssim_value + 1) / 2
        reward = max(min(reward, max_reward), min_reward)  # Clip to reward range
        
        rewards.append(ssim_value)
    
    return rewards

def reward_SVG_format(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, 
                      proper_format_reward=1.0, improper_format_penalty=-1.0, **kwargs):
    """
    Computes a reward based on whether the predicted SVG has proper opening and closing tags.
    
    This function checks if each predicted SVG string starts with an opening <svg tag 
    and ends with a closing </svg> tag. If both conditions are met, it returns a positive 
    reward; otherwise, it returns a penalty.
    
    Parameters
    ----------
    input_images : list
        List of input images. (Unused in this function)
    svg_rasters : list
        List of rasterized SVG images. (Unused)
    gt_svgs : list
        List of ground truth SVG strings. (Unused)
    predicted_svg_raws : list
        List of predicted SVG strings to check for proper formatting.
    predicted_svg_processed : list
        List of processed SVG predictions. (Unused)
    proper_format_reward : float, optional
        Reward value for properly formatted SVGs (default: 1.0)
    improper_format_penalty : float, optional
        Penalty value for improperly formatted SVGs (default: -1.0)
    **kwargs : dict
        Additional keyword arguments.
        
    Returns
    -------
    list
        A list of rewards, one per predicted SVG, with:
        - proper_format_reward for SVGs with proper opening and closing tags
        - improper_format_penalty for SVGs without proper opening and closing tags
    """
    rewards = []
    
    for svg in predicted_svg_raws:
        # Check if svg is None or empty
        if svg is None or not svg.strip():
            rewards.append(improper_format_penalty)
            continue
            
        # Check for opening <svg tag (allowing for attributes)
        has_opening = svg.strip().startswith('<svg')
        
        # Check for closing </svg> tag
        has_closing = svg.strip().endswith('</svg>')
        
        if has_opening and has_closing:
            rewards.append(proper_format_reward)
        else:
            rewards.append(improper_format_penalty)
    
    return rewards

# def reward_color_histogram(input_images, svg_rasters, gt_svgs, predicted_svg_raws, predicted_svg_processed, 
#                           max_reward=1.0, min_reward=0, white_svg_penalty=-1.0, 
#                           bins=16, white_weight=0.2, color_space='rgb', 
#                           comparison_method='intersection', **kwargs):
#     """
#     Computes a reward based on the similarity of color histograms between input images
#     and their SVG reconstructions, with reduced influence from white/background pixels.
    
#     Args:
#         input_images: List of input images (numpy arrays or PIL Images)
#         svg_rasters: List of rasterized SVG images
#         gt_svgs: List of ground truth SVG strings (unused)
#         predicted_svg_raws: List of predicted SVG strings (unused)
#         predicted_svg_processed: List of processed SVG predictions (unused)
#         max_reward: Maximum reward value (default: 1.0)
#         min_reward: Minimum reward value for valid comparisons (default: -1.0)
#         white_svg_penalty: Fixed penalty for all-white SVGs (default: -1.0)
#         bins: Number of bins per channel in histogram (default: 16)
#         white_weight: Weight to apply to white/background pixels (default: 0.2)
#         color_space: Color space to use for histogram ('rgb' or 'hsv')
#         comparison_method: Method to compare histograms ('intersection', 'correlation', 'chi-square', 'bhattacharyya')
#         **kwargs: Additional keyword arguments
    
#     Returns:
#         list: Color histogram similarity rewards for each image pair
#     """
#     rewards = []
    
#     # Map comparison method names to OpenCV comparison methods
#     cv_comparison_methods = {
#         'intersection': cv2.HISTCMP_INTERSECT,  # Higher values indicate better match
#         'correlation': cv2.HISTCMP_CORREL,      # Higher values indicate better match
#         'chi-square': cv2.HISTCMP_CHISQR,       # Lower values indicate better match
#         'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA  # Lower values indicate better match
#     }
    
#     # Select the comparison method (default to intersection if invalid)
#     cv_method = cv_comparison_methods.get(comparison_method.lower(), cv2.HISTCMP_INTERSECT)
#     # Flag if the selected method returns distance (lower=better) instead of similarity (higher=better)
#     is_distance_metric = cv_method in [cv2.HISTCMP_CHISQR, cv2.HISTCMP_BHATTACHARYYA]
    
#     for input_image, svg_raster in zip(input_images, svg_rasters):
#         # Convert PIL Images to numpy arrays if necessary (single conversion)
#         if not isinstance(input_image, np.ndarray):
#             input_image = np.array(input_image)
#         if not isinstance(svg_raster, np.ndarray):
#             svg_raster = np.array(svg_raster)
        
#         # Check if the SVG is all white (or nearly all white) - optimized detection
#         if len(svg_raster.shape) == 3:
#             # For RGB images, use more efficient vectorized operation
#             white_pixels_mask = np.all(svg_raster >= 250, axis=2)
#             non_white_pixels = np.sum(~white_pixels_mask)
#         else:
#             non_white_pixels = np.sum(svg_raster < 250)
            
#         total_pixels = svg_raster.size / (3 if len(svg_raster.shape) == 3 else 1)
        
#         # If less than 0.5% of pixels are non-white, consider it an all-white SVG
#         if non_white_pixels < total_pixels * 0.005:
#             rewards.append(white_svg_penalty)
#             continue
        
#         # Ensure images are in RGB format (with minimal conversions)
#         if len(input_image.shape) == 2:
#             input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
#         if len(svg_raster.shape) == 2:
#             svg_raster = cv2.cvtColor(svg_raster, cv2.COLOR_GRAY2RGB)
        
#         # Convert to the desired color space if not RGB
#         if color_space.lower() == 'hsv':
#             input_image_cs = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
#             svg_raster_cs = cv2.cvtColor(svg_raster, cv2.COLOR_RGB2HSV)
            
#             # Set appropriate ranges for HSV
#             channel_ranges = [180, 256, 256]  # OpenCV HSV ranges
#         else:  # default to RGB
#             input_image_cs = input_image
#             svg_raster_cs = svg_raster
#             channel_ranges = [256, 256, 256]  # RGB ranges
            
#         # Create mask for white pixels (based on original RGB)
#         white_mask_input = np.all(input_image >= 250, axis=2)
#         white_mask_svg = np.all(svg_raster >= 250, axis=2)
        
#         # Create weight masks (1.0 for normal pixels, white_weight for white pixels)
#         weight_mask_input = np.ones_like(white_mask_input, dtype=np.float32)
#         weight_mask_input[white_mask_input] = white_weight
        
#         weight_mask_svg = np.ones_like(white_mask_svg, dtype=np.float32)
#         weight_mask_svg[white_mask_svg] = white_weight
        
#         # Use OpenCV's calcHist for better performance
#         similarity_score = 0.0
        
#         # Process each channel with OpenCV functions
#         for channel in range(3):
#             # Extract channel data
#             input_channel = input_image_cs[:,:,channel]
#             svg_channel = svg_raster_cs[:,:,channel]
            
#             # Compute histograms with OpenCV
#             hist_input = cv2.calcHist([input_channel], [0], None, [bins], 
#                                      [0, channel_ranges[channel]])
#             hist_svg = cv2.calcHist([svg_channel], [0], None, [bins], 
#                                    [0, channel_ranges[channel]])
            
#             # Apply weights
#             for i in range(bins):
#                 bin_val_input = (i / bins) * channel_ranges[channel]
#                 bin_val_svg = (i / bins) * channel_ranges[channel]
                
#                 # Scale histogram bin values by white pixel weights
#                 # (This is an approximation - precise weighting would require custom histogram calculation)
#                 if bin_val_input >= 250:  # Bin contains white pixels
#                     hist_input[i] *= white_weight
#                 if bin_val_svg >= 250:  # Bin contains white pixels
#                     hist_svg[i] *= white_weight
            
#             # Normalize histograms for comparison
#             cv2.normalize(hist_input, hist_input, 0, 1, cv2.NORM_MINMAX)
#             cv2.normalize(hist_svg, hist_svg, 0, 1, cv2.NORM_MINMAX)
            
#             # Compare histograms using selected method
#             comparison = cv2.compareHist(hist_input, hist_svg, cv_method)
            
#             # For distance metrics, convert to similarity (1 - normalized distance)
#             if is_distance_metric:
#                 # Normalize to [0,1] range and invert
#                 comparison = 1.0 - min(comparison, 1.0)
            
#             # Add to total similarity
#             similarity_score += comparison / 3.0  # Average across channels
        
#         # Scale the similarity to reward range
#         reward = min_reward + (max_reward - min_reward) * similarity_score
#         # reward = max(min(reward, max_reward), min_reward)  # Clip to reward range
#         rewards.append(reward)
    
#     return rewards

def test():
    import os
    from PIL import Image
    # p = '/mnt/starvector/RL/logs/starvector/starvector-1b-im2svg_starvector-RL-20250306-032412/policy_outputs/step_7/9f93ecb99606bd971a3ed3a990233b5e04fc521f_repeat_3_process_3'
    # p= '/mnt/starvector/RL/logs/starvector/starvector-1b-im2svg_starvector-RL-20250306-032412/policy_outputs/step_7/9f93ecb99606bd971a3ed3a990233b5e04fc521f_repeat_4_process_3'
    # image_1 = Image.open(os.path.join(p, 'input_image.png')) 
    # image_2 = Image.open(os.path.join(p, 'svg_raster.png'))


    im1 = '/mnt/home/starvector-edit/star-vector-dev/assets/reward_assets/testim.png'
    im2 = '/mnt/home/starvector-edit/star-vector-dev/assets/reward_assets/tyestim2.png'
    image_1 = Image.open(im1)
    image_2 = Image.open(im2)

    # MSE = l2_distance(image_1, image_2, masked_l2=True)
    MSE = l2_distance(image_1, image_2, masked_l2=False, use_edge=True)

    l2_exp_decay = reward_L2_reconstruction([image_1], [image_2], [], [], [], sensitivity=5, masked=True, white_svg_penalty=-1.0)
    print(l2_exp_decay,MSE,sep=" OK ")

    # from starvector.data.util import rasterize_svg
    # import omegaconf
    # config = omegaconf.OmegaConf.load("star-vector-release/test-RL/train_config.yaml")
    # reward_functions = get_reward_function(config["rewards"])

    # gt_svg = """<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">\n<path fill-rule="evenodd" clip-rule="evenodd" d="M19.0909 1H4.90909C3.85636 1 3 1.8635 3 2.925V21.075C3 22.1362 3.85636 23 4.90909 23H19.0909C20.1434 23 21 22.1362 21 21.075V2.925C21 1.8635 20.1434 1 19.0909 1ZM19.3636 21.075C19.3636 21.2241 19.2387 21.35 19.0909 21.35H4.90909C4.76126 21.35 4.63636 21.2241 4.63636 21.075V2.925C4.63636 2.77594 4.76126 2.65 4.90909 2.65H7.90928V5.40001C7.90928 5.8557 8.27555 6.22501 8.72747 6.22501H15.2729C15.7248 6.22501 16.0911 5.8557 16.0911 5.40001V2.65H19.0909C19.2387 2.65 19.3636 2.77594 19.3636 2.925V21.075ZM14.4547 2.65H9.54565V4.57501H14.4547V2.65ZM14.4545 10.35H9.27273C8.82081 10.35 8.45454 10.7194 8.45454 11.175C8.45454 11.6307 8.82081 12 9.27273 12H14.4545C14.9065 12 15.2727 11.6307 15.2727 11.175C15.2727 10.7194 14.9065 10.35 14.4545 10.35ZM7.6367 13.65H16.364C16.8159 13.65 17.1822 14.0193 17.1822 14.475C17.1822 14.9306 16.8159 15.3 16.364 15.3H7.6367C7.18478 15.3 6.81852 14.9306 6.81852 14.475C6.81852 14.0193 7.18478 13.65 7.6367 13.65ZM16.364 16.95H7.6367C7.18478 16.95 6.81852 17.3193 6.81852 17.775C6.81852 18.2307 7.18478 18.6 7.6367 18.6H16.364C16.8159 18.6 17.1822 18.2307 17.1822 17.775C17.1822 17.3193 16.8159 16.95 16.364 16.95Z"/>\n</svg>"""
    # pred_svg = """<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">\n    <path d="M16.375 17.1a.825.825 0 1 1 0 1.65h-8.75a.825.825 0 1 1 0-1.65h8.75Zm0-3.3a.825.825 0 1 1 0 1.65h-8.75a.825.825 0 1 1 0-1.65h8.75Zm-2.45-3.3a.825.825 0 1 1 0 1.65h-5.3a.825.825 0 1 1 0-1.65h5.3Zm6.6-8.25c1.033 0 1.875.842 1.875 1.875v18.25c0 1.033-.842 1.875-1.875 1.875H5.075C4.042 23 3.2 22.158 3.2 21.125V2.875C3.2 1.842 4.042 1 5.075 1h13.85Zm-.325 1.875a.325.325 0 0 0-.325-.325H16.1v2.775c0.483-.391.875-.875.875h-6.6a.875.875 0 0 1-.875-.875V2.55H5.075a.325.325 0 0 0-.325.325v18.25c0.184.149.325.325.325h13.85a.325.325 0 0 0.325-.325V2.875ZM14.65 2.55h-4.9V4.4h4.9V2.55Z"/>\n</svg>"""

    # gt_im = rasterize_svg(gt_svg)
    # pred_im = rasterize_svg(pred_svg)

    # Plot and save the ground truth and predicted images side by side
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # # Create a figure with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # # Display ground truth image
    # ax1.imshow(np.array(gt_im))
    # ax1.set_title('Ground Truth')
    # ax1.axis('off')
    
    # # Display predicted image
    # ax2.imshow(np.array(pred_im))
    # ax2.set_title('Prediction')
    # ax2.axis('off')
    
    # # Add a main title
    # fig.suptitle('Comparison of Ground Truth and Predicted SVG Renders', fontsize=14)
    
    # # Adjust layout and save
    # plt.tight_layout()
    # plt.savefig('test.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print("Comparison image saved as 'test.png'")

    # masked_l2 = reward_L2_reconstruction_masked([gt_im], [pred_im], [], [], [])
    # dino_score = reward_dinoscore([gt_im], [pred_im], [], [], [])
    # print(masked_l2)

    # print(reward_token_len_deviation_v2([], [], [gt_svg], [pred_svg], [pred_svg]))

if __name__ == "__main__":
    test()