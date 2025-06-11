import imp
import re
from typing import Dict
from svg_util import robust_svg_to_pil, pil_png_to_np
import math
from rewarder import ImageRewarder, TextRewarder, ClipRewarder

import numpy as np
import traceback
import os
import yaml
from pprint import pprint 
from PIL import Image

class RewarderV2:
    def __init__(self, config_path=None):
        # get config path from env variable
        if config_path is None:
            self.config_path = os.environ.get("REWARD_CONFIG_PATH")
        else:
            self.config_path = config_path
        self.config = yaml.safe_load(open(self.config_path, "r"))
        pprint(self.config)
        if "image" in self.config:
            # Image - Image Reward
            self.img_rewarder = ImageRewarder(config=self.config["image"]['rewards'])
        if "text" in self.config:
            # Text - Text Reward
            self.text_rewarder = TextRewarder(config=self.config["text"]['rewards'])
        if "clip" in self.config:
            # Image - Text Reward
            self.clip_rewarder = ClipRewarder(config=self.config["clip"]['rewards'])

    def score_text_svg(self, text: str, svg: str) -> Dict[str, float]:
        MIN_REWARD = -1.0
        MAX_REWARD = 1.0
        RASTERIZE_SIZE = 512
        reward_final = {"overall": 0.0}
        try:
            img_pred, status = robust_svg_to_pil(svg, extract=True, repair=False)
            img_pred = pil_png_to_np(img_pred)
            if status == "invalid":
                return {"overall": -1.0, "renderable": 0}
            # white image
            if np.mean(img_pred) >= 254 or np.max(img_pred) - np.min(img_pred) < 5:
                return {"overall": -1.0, "renderable": 1, "non_white": 0}
            reward_clip = self.clip_rewarder(text, img_pred)
            reward_final['overall'] = reward_clip['clip_overall']
            reward_final.update(reward_clip)

        except Exception as e:
            traceback.print_exc()
            return {"overall": -1.0, "error": str(e)}
        # print(reward_final)
        return reward_final


    def score_svg_svg(self, svg_pred: str, svg_gt: str, img_gt: np.ndarray = None) -> Dict[str, float]:
        MIN_REWARD = -1.0
        MAX_REWARD = 1.0
        RASTERIZE_SIZE = 512

        reward_final = {"overall": 0.0}
        try:
            if img_gt is None:
                img_gt,status_gt = robust_svg_to_pil(svg_gt,RASTERIZE_SIZE,RASTERIZE_SIZE, extract=False, repair=False)
                img_gt = pil_png_to_np(img_gt)
            
            img_pred,status_pred = robust_svg_to_pil(svg_pred, RASTERIZE_SIZE, RASTERIZE_SIZE, extract=False, repair=False)
            img_pred = pil_png_to_np(img_pred)
            if status_pred == "invalid":
                return {"overall": -1.0, "renderable": 0}
            if np.mean(img_pred) >= 254 or np.max(img_pred) - np.min(img_pred) < 5:
                return {"overall": -1.0, "renderable": 1, "non_white": 0}

            # text rewards
            text_rewards = self.text_rewarder(svg_gt, svg_pred)
            reward_final['overall'] += text_rewards['text'] * self.config["text"]['weight']
            reward_final.update(text_rewards)
            # img rewards
            img_rewards = self.img_rewarder(img_gt, img_pred)
            reward_final['overall'] += img_rewards['image'] * self.config["image"]['weight']
            reward_final.update(img_rewards)

            return reward_final
        except Exception as e: # data error
            # breakpoint()
            traceback.print_exc()
            return {"overall": -1.0, "error": str(e)}
    
