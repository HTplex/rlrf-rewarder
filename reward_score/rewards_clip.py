from PIL import Image
from sentence_transformers import SentenceTransformer, util
import numpy as np
import math

class RewardClip:
    def __init__(self):
        self.model = None
    def __call__(self, text, img, model_name):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.model is None:
            self.model = SentenceTransformer(model_name)
        img_emb = self.model.encode(img, convert_to_tensor=True)
        text_emb = self.model.encode(text, convert_to_tensor=True)
        cos_sim = util.cos_sim(img_emb, text_emb).item()
        return cos_sim

from vllm import LLM, SamplingParams
from PIL import Image
import math
import random
from pprint import pprint
import json, re

def extract_json(reply: str, *, return_dict: bool = True):
    """
    Return the first well-formed JSON object found in an LLM reply.

    Parameters
    ----------
    reply : str
        Raw LLM output (may mix prose, code fences, etc.).
    return_dict : bool, default True
        True  → parsed Python object
        False → raw JSON string

    Returns
    -------
    dict | list | str | None
        Parsed JSON (or its string) if found; otherwise None.
    """

    # 1) Try fenced or bare JSON via regex
    pattern = re.compile(
        r"""```(?:json)?\s*({.*?})\s*```   # ```json … ```
          |                               #   or
            ({.*})                        # bare JSON
        """,
        re.DOTALL | re.VERBOSE,
    )
    for m in pattern.finditer(reply):
        candidate = m.group(1) or m.group(2)
        try:
            parsed = json.loads(candidate)
            return parsed if return_dict else candidate
        except json.JSONDecodeError:
            continue  # keep scanning

    # 2) Fallback: find first balanced { … }
    start = reply.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(reply[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = reply[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed if return_dict else candidate
                    except json.JSONDecodeError:
                        break

    return None


class QwenGeo:
    def __init__(self):
        self.system_prompt = '<|im_start|>system\n你是一个助手<|im_end|>\n'
        self.prompt_geo = """
            <|im_start|>user
            <|vision_start|><|image_pad|><|vision_end|> 
                你是一位严谨的几何教师兼竞赛评审，同时也是图形美观性评委。  
                给定一段几何题描述（文本）和一幅候选示意图（图像），你必须依次完成：

                ◇ 步骤 1：文本解析  
                - 识别并枚举题目出现的所有几何元素：点、线段、直线、圆、多边形等。  
                - 提取它们之间的几何关系（如“共线、垂直、平行、等长、等角、位于圆上、顺序排列”等）。  
                - 以 **relation_list** 数组形式记录，例如：  
                    • "A,B,C 共线并按此顺序"  
                    • "∠ABC = ∠DEF"  
                    • "AB = AC"  
                    • "点 D 在圆 O 上"  
                如果文本中有隐含关系也要推断并列出（但请标注为 "implicit"）。

                ◇ 步骤 2：图像核对  
                对 relation_list 中的每一条关系，检查图像是否满足，产出布尔结果 `true/false`。  

                ◇ 步骤 3：打分  
                1. **几何准确度 geometry_score (0–10，可用小数)**  
                    - 先按下表确定所属整数档位，再在该档内细调：  
                    10 完美；9 几乎完美；8 轻微误差；7 小量错误；6 中等错误；  
                    5 显著错误；4 严重错误；3 极严重；2 近乎无关；1 虚假图形；0 不可用。  
                    - 评估指标：relation_list 的命中率、标签对应、整体结构完整性。  

                2. **美观评分 aesthetics_score (0–5，可用小数)**  
                    5 极佳：线条清晰、布局平衡、标签易读、无多余元素；  
                    4 良好：轻微凌乱或线宽不一；  
                    3 中等：明显杂乱或比例失衡；  
                    2 较差：线条粗糙、元素重叠；  
                    1 很差：几乎无法辨认；  
                    0 无法评价（空白或严重损坏）。  
                    - 若图中包含大段题目原文或无关文字，应 **至少扣 1 分**，并在解释中说明。  

                ◇ 输出  
                只输出 **以下 JSON**，不要添加多余键、Markdown 或代码块：

                {
                "geometry": {
                    "score": <float 0-10>,
                    "explanation": "<≤40 字概述主要原因>",
                    "relations_checked": [
                    {"relation": "<字符串>", "result": true | false},
                    ...
                    ]
                },
                "aesthetics": {
                    "score": <float 0-5>,
                    "explanation": "<≤40 字概述主要原因>"
                }
                }
                
                题目：
                
            """
        self.llm = None
    def __call__(self, text: str, img: Image.Image, model_name: str, method: str = "draw", size:int=-1) -> float:
        if self.llm is None:
            self.llm = LLM(model=model_name, max_model_len=4096, trust_remote_code=True)

        prompt = self.system_prompt + self.prompt_geo + text + "\n<|im_end|>\n<|im_start|>assistant\n"
        size = 512
        # resize keep ratio
        w, h = img.size
        if w > h:
            img = img.resize((size, int(size*h/w)))
        else:
            img = img.resize((int(size*w/h), size))
        # img = img.resize((size, size))
        vllm_inputs = [{"prompt": prompt, "multi_modal_data": {"image": [img]}}]
        sampling_params = SamplingParams(
            temperature=0.5, 
            max_tokens=2048,
            )

        completions = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        raw_output = completions[0].outputs[0].text
        
        output_json = extract_json(raw_output)
        print(output_json)
        try:
            # geo_score = output_json['geometry']['score']/5-1
            geo_scores = []
            for relation in output_json['geometry']['relations_checked']:
                if relation['result']:
                    geo_scores.append(1)
                else:
                    geo_scores.append(0)
            if len(geo_scores) == 0:
                geo_score = 0
            else:
                geo_score = np.mean(geo_scores)
            geo_score = min(geo_score*10, float(output_json['geometry']['score']))
            geo_score = geo_score/5-1
            print(geo_score)
            aesthetics_score = output_json['aesthetics']['score']/2.5-1
            return 0.8*geo_score + 0.2*aesthetics_score
        except:
            return 0

        
        

class QwenClip:
    def __init__(self):
        self.system_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        self.prompt_draw = '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Does the drawing resemble the description: "{}" [Yes/No]<|im_end|>\n<|im_start|>assistant\n'
        self.prompt_image_accurate = '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Does the image match the description clearly, accurately, and aesthetically pleasing: "{}" [Yes/No]<|im_end|>\n<|im_start|>assistant\n'
        
        self.llm = None
    def __call__(self, text: str, img: Image.Image, model_name: str, method: str = "draw", size:int=-1) -> float:
        if self.llm is None:
            self.llm = LLM(model=model_name, trust_remote_code=True)

        if method == "draw":
            prompt = self.prompt_draw.format(text)
        elif method == "image":
            prompt = self.prompt_image.format(text)
        elif method == "image_accurate":
            prompt = self.prompt_image_accurate.format(text)
        else:
            raise ValueError(f"Invalid method: {method}")
        prompt = self.system_prompt + prompt
        if size == -1:
            size = random.choice([384, 444, 512, 696, 768])
        # resize keep ratio
        w, h = img.size
        if w > h:
            img = img.resize((size, int(size*h/w)))
        else:
            img = img.resize((int(size*w/h), size))
        # img = img.resize((size, size))
        vllm_inputs = [{"prompt": prompt, "multi_modal_data": {"image": [img]}}]
        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=10,
            logprobs=20,                 # ← request log-probs
            # repetition_penalty=0.5
            )

        completions = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        output = completions[0].outputs[0]
        # 
        # Instead of iterating over a non-existent "tokens" attribute, use "output_token_ids"
        try:
            yes_prob = math.exp(output.logprobs[0][9454].logprob)
        except:
            yes_prob = 0

        return yes_prob*2-1





