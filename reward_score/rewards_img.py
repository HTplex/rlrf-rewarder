import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import cv2, numpy as np
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from starvector2.utils.svg_util import show_img_np

# -------------------------------------------------------------
def _foreground_mask(img: np.ndarray, thresh: int = 0.96) -> np.ndarray:
    """
    Binary mask of 'non‑white' pixels.
    Returns an array with shape (H, W, 1) and values ∈ {0, 1}.
    """
    if img.ndim == 3:                    # RGB / BGR / RGBA
        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    else:                                # already gray
        gray = img
    return (gray < thresh).astype(np.float32)[..., None]   # broadcast‑ready
# -------------------------------------------------------------

def mse_metric(img_label, img_pred):
    image1_tensor = ToTensor()(img_label)
    image2_tensor = ToTensor()(img_pred)
    mse = F.mse_loss(image1_tensor, image2_tensor)
    return mse.item()

def reward_mse(
        img_label: np.ndarray,
        img_pred : np.ndarray,
        *,
        use_mask   : bool   = True,      # foreground‑weighted or plain MSE
        bg_weight  : float  = 0.0,       # weight assigned to background pixels
        exp_reward : bool   = True,      # map MSE → reward with exp(‑k·MSE)
        k          : float  = 3.0,       # steepness of exp mapping
    ) -> float:
    # if os.environ.get("DEBUG", "false") == "true":
    #     breakpoint()
    """
    Parameters
    ----------
    img_label : ndarray  – ground‑truth raster
    img_pred  : ndarray  – raster rendered from the SVG candidate
    use_mask  : bool     – if True, background contributes `bg_weight` to the loss
    bg_weight : float    – importance of background (0 = ignore, 1 = full)
    exp_reward: bool     – if True, reward = exp(‑k·mse); else reward = 1‑mse
    k         : float    – controls sharpness of exp() mapping

    Returns
    -------
    reward : float       – higher is better, in (0, 1]
    """
    # 1) resize & drop alpha -----------------------------------------------
    h, w = img_label.shape[:2]
    img_pred = cv2.resize(img_pred, (w, h), interpolation=cv2.INTER_AREA)
    img_label = img_label[..., :3]
    img_pred  = img_pred [..., :3]

    # 2) convert to 0‑1 floats ---------------------------------------------
    img_label = img_label.astype(np.float32) / 255.0
    img_pred  = img_pred .astype(np.float32) / 255.0

    # 3) squared error -----------------------------------------------------
    err = (img_label - img_pred) ** 2

    # 4) optional foreground weighting -------------------------------------
    if use_mask:        
        w_fg = _foreground_mask(img_label) + _foreground_mask(img_pred)
        weights = bg_weight + (1.0 - bg_weight) * w_fg       # 0.1 … 1.0
        weighted_err = err * weights
        mse = weighted_err.sum() /(weights.sum() + 1e-6)             # ← normalise by ∑w
    else:
        mse = err.mean()

    # 5) map to reward ------------------------------------------------------
    if exp_reward:
        reward = float(np.exp(-k * mse))           #  (0, 1]
    else:
        reward = float(np.clip(1.0 - mse, 0.0, 1.0))
    return reward

def reward_img_l2(img_label, img_pred, canny=False, dilate=True, dilate_kernel_size=3, dilate_iterations=1, gray=False, blur=0, normalize="avg_std"):
    """
    Compute the L2 loss between img_label and img_pred.
    
    Parameters:
        img_label: Ground truth image.
        img_pred: Predicted image.
        gray (bool): Convert images to grayscale if True.
        blur (int): Size of Gaussian blur kernel; if > 0, apply blur.
        normalize (str): If set to "avg_std", normalize each image using its mean and std;
                         if truthy (but not "avg_std"), normalize by dividing by 255.
    
    Returns:
        l2: The mean squared difference (L2 loss) between the processed images.
    """
    # Resize predicted image to match label image dimensions
    img_pred = cv2.resize(img_pred, (img_label.shape[1], img_label.shape[0]))
    if canny:
        img_pred = canny_img(img_pred, dilate=dilate, dilate_kernel_size=dilate_kernel_size, dilate_iterations=dilate_iterations)
        img_label = canny_img(img_label, dilate=dilate, dilate_kernel_size=dilate_kernel_size, dilate_iterations=dilate_iterations)


    # Handle mismatched channels (e.g., RGBA vs RGB)
    if img_label.ndim == 3 and img_label.shape[2] == 4:
        img_label = img_label[:, :, :3]
    if img_pred.ndim == 3 and img_pred.shape[2] == 4:
        img_pred = img_pred[:, :, :3]
        
    if blur > 0:
        img_label = cv2.GaussianBlur(img_label, (blur, blur), 0)
        img_pred = cv2.GaussianBlur(img_pred, (blur, blur), 0)

    if gray:
        img_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    if normalize:
        # Normalize images to -1 to 1 range
        if normalize == "avg_std":
            # Convert images to float and normalize using their own average and std
            img_label = img_label.astype(np.float32)
            img_pred = img_pred.astype(np.float32)
            img_label = (img_label - np.mean(img_label)) / np.std(img_label)
            img_pred = (img_pred - np.mean(img_pred)) / np.std(img_pred)
            l2 = np.sum((img_label - img_pred) ** 2) / np.prod(img_label.shape)
            if np.isnan(l2):
                reward = -1.0
            else:
                reward = -1 * np.clip(l2-1, -1, 1)

        else:
            # Fallback: normalize 
            img_label = img_label.astype(np.float32) / 255.0 
            img_pred = img_pred.astype(np.float32) / 255.0
            l2 = np.sum((img_label - img_pred) ** 2) / np.prod(img_label.shape)
            reward = -1 * np.clip(l2*2-1, -1, 1)

    
    # Compute the mean squared error between the images
    return reward


def reward_ssim(img_label, img_pred, max_reward=1.0, min_reward=-1.0, win_size=11, 
               use_grayscale=False):
    """
    Compute the SSIM between img_label and img_pred.
    
    Parameters:
        img_label: Ground truth image.
        img_pred: Predicted image.
    """
    # Resize predicted image to match label image dimensions
    img_pred = cv2.resize(img_pred, (img_label.shape[1], img_label.shape[0]))
    # Check if inputs are numpy arrays
    if not isinstance(img_label, np.ndarray):
        img_label = np.array(img_label)
    if not isinstance(img_pred, np.ndarray):
        img_pred = np.array(img_pred)
    
    # Calculate SSIM
    try:
        if use_grayscale:
            # Convert to grayscale if requested
            if len(img_label.shape) == 3 and img_label.shape[2] == 3:
                img_label_gray = np.mean(img_label, axis=2).astype(np.uint8)
            else:
                img_label_gray = img_label
                
            if len(img_pred.shape) == 3 and img_pred.shape[2] == 3:
                img_pred_gray = np.mean(img_pred, axis=2).astype(np.uint8)
            else:
                img_pred_gray = img_pred
            
            ssim_value = ssim(img_label_gray, img_pred_gray, 
                              data_range=255, win_size=win_size)
        else:
            # Use multichannel SSIM for RGB images
            multichannel = len(img_label.shape) == 3 and img_label.shape[2] == 3
            if multichannel:
                ssim_value = ssim(img_label, img_pred, 
                                 data_range=255, win_size=win_size, 
                                 channel_axis=2)
            else:
                ssim_value = ssim(img_label, img_pred, 
                                 data_range=255, win_size=win_size)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return min_reward
        
    # Map SSIM value ([-1, 1]) to reward range ([min_reward, max_reward])
    # SSIM values are typically in [0, 1] range for realistic images
    reward = min_reward + (max_reward - min_reward) * (ssim_value + 1) / 2
    reward = np.clip(reward, min_reward, max_reward)  # Clip to [-1,1]
    
    return reward


def white_image_penalty(img_label, img_pred, threshold=250, min_nonwhite_percentage=0.005, penalty=-1.0):
    # Check if the predicted image is all white (or nearly all white)
    if len(img_pred.shape) == 3:
        # For RGB images, check if all channels are close to white
        non_white_pixels = np.sum(np.any(img_pred < threshold, axis=2))
    else:
        non_white_pixels = np.sum(img_pred < 250)
        
    total_pixels = img_pred.shape[0] * img_pred.shape[1]
    
    # If less than 0.5% of pixels are non-white, consider it an all-white image
    if non_white_pixels < total_pixels * min_nonwhite_percentage:
        return penalty
    else:
        return 0.0
    

import numpy as np
import cv2
import torch
from DISTS_pytorch import DISTS
from dreamsim import dreamsim

def dists_score(img_gt: np.ndarray,
                img_pred:   np.ndarray,
                device: str = "auto") -> float:
    """
    Compute the Deep Image Structure & Texture Similarity (DISTS) score
    between two OpenCV images.

    Parameters
    ----------
    img_pred, img_gt : np.ndarray (H, W, 3) or (H, W) uint8
        Images in BGR colour space as returned by cv2.*.
    device : {"cuda", "cpu"}, default "cuda"
        Where to run the metric.

    Returns
    -------
    float
        DISTS score – higher means more similar (Max ≈ 1.0).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB float32 tensor in [0, 1] shaped (1, 3, H, W)."""
        if img.ndim == 2:                                  # grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:                                              # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0               # → [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1)    # (3,H,W)
        return tensor.unsqueeze(0).to(device)              # (1,3,H,W)

    pred_tensor = _to_tensor(img_pred)
    gt_tensor   = _to_tensor(img_gt)

    metric = DISTS().to(device)

    with torch.no_grad():                                  # drop this for training
        score = metric(pred_tensor, gt_tensor).item()


    return score

def dist_reward(img_gt: np.ndarray, img_pred: np.ndarray) -> float:
    return 1-dists_score(img_gt, img_pred)*2

import cv2
import numpy as np
import torch
from PIL import Image
from dreamsim import dreamsim
from os.path import expanduser, join, dirname


class DreamSimDistanceReward:
    """
    Wrapper around DreamSim that loads the model once and exposes
    a callable interface for computing perceptual distance.
    
    Example
    -------
    >>> metric = DreamSimDistance(device="cuda")
    >>> dist   = metric(img_pred, img_gt)   # lower → more similar
    """

    def __init__(self, device: str = "auto"):
        """
        Parameters
        ----------
        device : {"cuda", "cpu"}, default "cpu"
            Device on which the DreamSim model runs.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None


    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV BGR image (H×W×C uint8 or H×W uint8) to a tensor
        accepted by DreamSim: (1, 3, 224, 224), float in [0, 1].
        """
        if img_bgr.ndim == 2:  # grayscale → RGB
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:  # BGR → RGB
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_bgr)
        tensor = self.preprocess(pil_img).to(self.device)  # (C,H,W) or (1,C,H,W)
        if tensor.ndim == 3:  # make batch dimension explicit
            tensor = tensor.unsqueeze(0)
        return tensor

    @torch.no_grad()
    def __call__(self,
                 img_gt: np.ndarray,
                 img_pred:   np.ndarray,
                 model_path: str = None,
                 canny: bool = False,
                 dilate: bool = False,
                 dilate_kernel_size: int = 3,
                 dilate_iterations: int = 1) -> float:
        """
        Compute DreamSim perceptual distance between two images.

        Returns
        -------
        float
            Distance (lower → images are more similar).
        """
        if self.model is None:
            print(f"!!!!!!!!!! DreamSim model_path: {model_path}!!!!!!!!!!")
            self.model, self.preprocess = dreamsim(pretrained=True, device=self.device, cache_dir=model_path)
            self.model.eval()  # inference mode
        if canny:
            img_pred = canny_img(img_pred, dilate=dilate, dilate_kernel_size=dilate_kernel_size, dilate_iterations=dilate_iterations)
            img_gt = canny_img(img_gt, dilate=dilate, dilate_kernel_size=dilate_kernel_size, dilate_iterations=dilate_iterations)
        t_pred = self._to_tensor(img_pred)
        t_gt   = self._to_tensor(img_gt)
        loss = self.model(t_pred, t_gt).item()
        return 1 - loss*2
    

def canny_img(img: np.ndarray,
              threshold1: int = 50,
              threshold2: int = 150,
              dilate: bool = False,
              dilate_kernel_size: int = 3,
              dilate_iterations: int = 1) -> np.ndarray:
    """
    Run Canny edge detection on `img`, binarise the output (edges = 255),
    and optionally dilate the resulting edge map.

    The returned image is a 3-channel, 8-bit (uint8) OpenCV image
    (H×W×3, BGR order) so it can be used wherever a colour image
    is expected.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR or grayscale uint8 format.
    threshold1, threshold2 : int, default (50, 150)
        Hysteresis thresholds for cv2.Canny.
    dilate : bool, default False
        If True, dilate the edge map.
    kernel_size : int, default 3
        Size of the square kernel used for dilation.
    iterations : int, default 1
        Number of dilation iterations.

    Returns
    -------
    np.ndarray
        3-channel uint8 edge map (BGR) where edge pixels are white (255,255,255)
        and background pixels are black (0,0,0).
    """
    # 1. Convert to grayscale if necessary
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # 3. Binarise: any non-zero value → 255
    edges[edges != 0] = 255

    # 4. Optional dilation to thicken edges
    if dilate:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    # 5. Convert single-channel edge map to 3-channel (BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_bgr

from lpips import LPIPS
import torch
import cv2

class RewardLpips:
    """Compute LPIPS similarity and map it to a reward in [-1, 1].

    This helper now automatically moves the LPIPS network and the input tensors to GPU
    when a CUDA device is available (unless the environment variable `LPIPS_CPU_ONLY=1`
    is set).
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"!!!!!!!!!! LPIPS device: {self.device}!!!!!!!!!!")
        
        self.lpips = LPIPS(net="alex").to(self.device)

    def __call__(self, img_label, img_pred, canny: bool = False):
        # show_img_np(img_label)
        # show_img_np(img_pred)
        if canny:
            img_pred = canny_img(img_pred)
            img_label = canny_img(img_label)
        # Ensure both images are 256×256 then convert to torch tensor in CHW format
        img_label = cv2.resize(img_label, (256, 256))
        img_pred = cv2.resize(img_pred, (256, 256))

        # BGR → RGB, HWC → CHW, 0-255 → 0-1
        img_label = (
            torch.from_numpy(img_label[..., ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        img_pred = (
            torch.from_numpy(img_pred[..., ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(self.device)
        )

        img_label = img_label.unsqueeze(0)
        img_pred = img_pred.unsqueeze(0)

        with torch.no_grad():
            lpips_score = self.lpips(img_label, img_pred)

        # LPIPS returns distance (0 identical); convert to similarity-style reward in [-1, 1]
        reward = (1.0 - 2.0 * lpips_score).clamp(-1.0, 1.0)
        return reward.squeeze().item()