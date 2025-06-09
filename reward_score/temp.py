svg1 = """
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 124 124" fill="none">
<rect width="124" height="124" rx="24" fill="#F97316"/>
<path d="M19.375 36.7818V100.625C19.375 102.834 21.1659 104.625 23.375 104.625H87.2181C90.7818 104.625 92.5664 100.316 90.0466 97.7966L26.2034 33.9534C23.6836 31.4336 19.375 33.2182 19.375 36.7818Z" fill="white"/>
<circle cx="63.2109" cy="37.5391" r="18.1641" fill="black"/>
</svg>
"""
svg2 = """
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 124 124" fill="none">
<rect width="124" height="124" rx="24" fill="#F97316"/>
<path d="M19.375 36.7818V100.625C19.375 102.834 21.1659 104.625 23.375 104.625H87.2181C90.7818 104.625 92.5664 100.316 90.0466 97.7966L26.2034 33.9534C23.6836 31.4336 19.375 33.2182 19.375 36.7818Z" fill="white"/>
<circle cx="63.2109" cy="37.5391" r="18.1641" fill="black"/>
</svg>
"""
from svg_util import robust_svg_to_pil, pil_png_to_np
from svg_rewarder import SVGRewarderV2
import cv2
from pprint import pprint
img, status = robust_svg_to_pil(svg1, extract=False, repair=True)
img_np = pil_png_to_np(img)
print(status)
# save img
img.save("../tmp/test.png")
cv2.imwrite("../tmp/test1.png", img_np)

svg_rewarder = SVGRewarderV2(config_path="./configs/ht_exp2_clip.yaml")
clip_score = svg_rewarder.clip_compute_score(text="a red circle", svg=svg1)
pprint(clip_score)

svg_rewarder = SVGRewarderV2(config_path="./configs/ht_exp2.yaml")
clip_score = svg_rewarder.svg_compute_score(svg_pred=svg1, svg_gt=svg2)
pprint(clip_score)