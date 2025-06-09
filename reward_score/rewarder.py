from rewards_svg import reward_svg_length_quadratic, reward_svg_length_juan, reward_svg_edit_distance, reward_aspect_ratio
from rewards_img import reward_img_l2, reward_ssim, white_image_penalty, reward_mse,dist_reward, DreamSimDistanceReward, RewardLpips, mse_metric
from rewards_clip import RewardClip, QwenClip, QwenGeo
import yaml
import os

class ImageRewarder:
    def __init__(self, config):
        self.config = config
        self.rewards = {
            "img_l2": reward_img_l2,
            "ssim": reward_ssim,
            "white_image_penalty": white_image_penalty,
            "mse": reward_mse,
            "mse_metric": mse_metric,

            "dist": dist_reward,
            "lpips": RewardLpips(),
            "dreamsim": DreamSimDistanceReward(),
        }

    def __call__(self, img_gt, img_pred):
        reward = {"image": 0.0}
        for reward_name, reward_info in self.config.items():
            reward_args = reward_info.get("args", {})
            reward_func = self.rewards[reward_info["func"]]
            reward[reward_name] = reward_func(
                img_gt, img_pred, **reward_args
            )
            reward["image"] += reward[reward_name] * reward_info.get("weight")
               

        return reward
    
class TextRewarder:
    def __init__(self, config):
        self.config = config
        self.rewards = {
            "svg_length_quadratic": reward_svg_length_quadratic,
            "aspect_ratio": reward_aspect_ratio,
        }

    def __call__(self, svg_gt, svg_pred):
        # todo parameterise using yaml
        reward = {"text": 0.0}

        for reward_name, reward_info in self.config.items():
            reward_args = reward_info.get("args", {})
            reward_func = self.rewards[reward_info["func"]]
            reward[reward_name] = reward_func(
                svg_gt, svg_pred, **reward_args
            )
            reward["text"] += reward[reward_name] * reward_info.get("weight")
               
        return reward

class ClipRewarder:
    def __init__(self, config):
        self.config = config
        self.rewards = {
            "clip": RewardClip(),
            "qwen_clip": QwenClip(),
            "qwen_geo": QwenGeo(),
        }

    def __call__(self, text, image):
        # todo parameterise using yaml
        reward = {"clip_overall": 0.0}
        for reward_name, reward_info in self.config.items():
            reward_args = reward_info.get("args", {})
            reward_func = self.rewards[reward_info["func"]]
            reward[reward_name] = reward_func(
                text, image, **reward_args
            )
            reward["clip_overall"] += reward[reward_name] * reward_info.get("weight")
               
        return reward    


