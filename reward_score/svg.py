import re
from typing import Dict
from starvector2.utils.svg_util import get_svg_original_size, post_process_svg, rasterize_svg,strip_svg_text, extract_svg, svg_to_np, strip_inline_elements
import math
from .rewarder import ImageRewarder, TextRewarder, ClipRewarder

import numpy as np
import traceback
import os
import yaml
from pprint import pprint 
from PIL import Image

class SVGRewarder:
    def __init__(self, config_path=None):
        # get config path from env variable
        if config_path is None:
            self.config_path = os.environ.get("REWARD_CONFIG_PATH")
        else:
            self.config_path = config_path
        self.config = yaml.safe_load(open(self.config_path, "r"))
        pprint(self.config)
        if "image" in self.config:
            self.img_rewarder = ImageRewarder(config=self.config["image"]['rewards'])
        if "text" in self.config:
            self.text_rewarder = TextRewarder(config=self.config["text"]['rewards'])
        if "clip" in self.config:
            self.clip_rewarder = ClipRewarder(config=self.config["clip"]['rewards'])

    def clip_compute_score(self, text: str, svg: str) -> Dict[str, float]:
        MIN_REWARD = -1.0
        MAX_REWARD = 1.0
        RASTERIZE_SIZE = 512
        reward_final = {"overall": 0.0}
        try:
            svg_no_think = strip_inline_elements(svg, [["<think>", "</think>"]])
            svg_extracted = extract_svg(svg_no_think)
            # svg_striped_text = strip_svg_text(svg_extracted)
            svg_extracted = post_process_svg(svg_extracted)['svg']
            img_pred = rasterize_svg(svg_extracted, RASTERIZE_SIZE, RASTERIZE_SIZE)
            # img_pred = svg_to_np(svg_extracted, RASTERIZE_SIZE, RASTERIZE_SIZE)
            # img_pred = Image.fromarray(img_pred).convert("RGB")
            # if pure white, return 0
            reward_clip = self.clip_rewarder(text, img_pred)
            reward_final['overall'] = reward_clip['clip_overall']
            reward_final.update(reward_clip)
            reward_final['white'] = 1.0
            if np.mean(img_pred) >= 254:
                reward_final['overall'] = -1.0
                reward_final['white'] = -1.0
            if np.max(img_pred) - np.min(img_pred) < 5:
                reward_final['overall'] = -1.0
                reward_final['white'] = -1.0
                # print("!!!!!!!!! begin white svg !!!!!!!!")
                # print("=======svg=========")
                # print(svg)
                # print("=======strip_inline=========")
                # print(svg_no_think)
                # print("=======extract=========")
                # print(svg_extracted)
                # print("=======strip_text=========")
                # print(svg_striped_text)
                # print("=======post_process=========")
                # print(svg_post_processed)
                # print("================")
                # print("!!!!!!!!! end white svg !!!!!!!!")
        except Exception as e:
            traceback.print_exc()
            return {"overall": -1.0, "error": str(e)}
        # print(reward_final)
        return reward_final


    def svg_compute_score(self, predict_str: str, ground_truth: str, img_gt: np.ndarray = None) -> Dict[str, float]:
        MIN_REWARD = -1.0
        MAX_REWARD = 1.0
        RASTERIZE_SIZE = 512
        svg_pred = predict_str
        svg_gt = ground_truth
        reward_final = {"overall": 0.0}
        try:
            res_post_process = post_process_svg(svg_pred)
            svg_pred = res_post_process['svg']
            svg_syntax_correct = not res_post_process['post_processed']
            svg_syntax_fixable = not res_post_process['no_compile']
            # calculate rasterize size
            if img_gt is None:
                w_gt,h_gt = get_svg_original_size(svg_gt)
            else:
                w_gt,h_gt = img_gt.shape[1], img_gt.shape[0]
            # compute new dimensions preserving aspect ratio: long edge = RASTERIZE_SIZE
            if w_gt >= h_gt:
                new_w = RASTERIZE_SIZE
                new_h = max(1, int(math.ceil(h_gt * RASTERIZE_SIZE / w_gt)))
            else:
                new_h = RASTERIZE_SIZE
                new_w = max(1, int(math.ceil(w_gt * RASTERIZE_SIZE / h_gt)))
            if img_gt is None:
                img_gt = np.array(rasterize_svg(svg_gt, new_h, new_w))
                # pad white to RASTERIZE_SIZE x RASTERIZE_SIZE
                img_gt_new = np.ones((RASTERIZE_SIZE, RASTERIZE_SIZE, 3), dtype=np.uint8) * 255
                img_gt_new[:img_gt.shape[0], :img_gt.shape[1], :] = img_gt
                img_gt = img_gt_new

            try:
                w_pred,h_pred = get_svg_original_size(svg_pred)
                # if w_pred is None or h_pred is None:
                #     w_pred, h_pred = w_gt, h_gt
                if w_pred >= h_pred:
                    new_w = RASTERIZE_SIZE
                    new_h = max(1, int(math.ceil(h_pred * RASTERIZE_SIZE / w_pred)))
                else:
                    new_h = RASTERIZE_SIZE
                    new_w = max(1, int(math.ceil(w_pred * RASTERIZE_SIZE / h_pred)))
                img_pred = np.array(rasterize_svg(svg_pred, new_h, new_w))
                img_pred_new = np.ones((RASTERIZE_SIZE, RASTERIZE_SIZE, 3), dtype=np.uint8) * 255
                img_pred_new[:img_pred.shape[0], :img_pred.shape[1], :] = img_pred
                img_pred = img_pred_new

                svg_renderable = True
            except:
                img_pred = np.ones_like(img_gt) * 255
                svg_renderable = False
            
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
    
