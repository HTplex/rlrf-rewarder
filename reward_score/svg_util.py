from PIL import Image
from bs4 import BeautifulSoup
from typing import Tuple
import re
from svgpathtools import svgstr2paths
import numpy as np
import cairosvg
from io import BytesIO
import xml.etree.ElementTree as ET
import random
import io
import cv2

def get_error_img(h=512, w=512):
    """
    Returns a white image of shape (h, w, 3) with a red 'X' in the center.
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    thickness = max(1, h // 50)
    color = (255, 0, 0)  # Red in RGB
    center_x, center_y = w // 2, h // 2
    offset = min(h, w) // 5
    cv2.line(img, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), color, thickness)
    cv2.line(img, (center_x - offset, center_y + offset), (center_x + offset, center_y - offset), color, thickness)
    return img


def robust_svg_to_pil(svg_string, output_width=512, output_height=512, extract=True, repair=False, failed_fallback="error"):
    if extract and repair:
        raise ValueError("extract and repair cannot be True at the same time")
    # construct fallback image
    if failed_fallback == "error":
        fallback_img = get_error_img(output_height, output_width)
    elif failed_fallback == "white":
        fallback_img = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
    elif failed_fallback == "black":
        fallback_img = np.ones((output_height, output_width, 3), dtype=np.uint8) * 0
    else:
        fallback_img = get_error_img(output_height, output_width)
    fallback_img = Image.fromarray(fallback_img)
    # extract svg from other crap
    if extract:
        svg_string = extract_svg(svg_string)
    # check and fix svg
    try:
        svgstr2paths(svg_string)
        svg_status = "valid"
    except Exception as e:
        svg_status = "invalid"
    # repair svg if invalid
    if svg_status == "invalid" and repair:
        svg_string = repair_svg(svg_string)
        try:
            svgstr2paths(svg_string)
            svg_status = "repaired"
        except Exception as e:
            svg_status = "invalid"
    if svg_status == "invalid":
        return fallback_img, svg_status
    # rasterize svg
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=output_width, output_height=output_height,
            dpi=128, scale=2)
        return Image.open(BytesIO(svg_raster_bytes)), svg_status
    except Exception as e:
        return fallback_img, "invalid"

    

def repair_svg(svg_text, output_width=None, output_height=None, fallback_svg="white"):
    if fallback_svg == "white":
        place_holder_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"></svg>"""
    
    soup = BeautifulSoup(svg_text, 'xml') # Read as soup to parse as xml
    svg_bs4 = soup.prettify() # Prettify to get a string

    # Store the original signal handler
    import signal
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set a timeout to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Try direct conversion without BeautifulSoup
        svg_cairo = cairosvg.svg2svg(svg_bs4, output_width=output_width, output_height=output_height).decode()
        
    except TimeoutError:
        print("SVG conversion timed out, using fallback method")
        svg_cairo = place_holder_svg
    finally:
        # Always cancel the alarm and restore original handler, regardless of success or failure
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        
    svg_clean = "\n".join([line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]) # Remove xml header
    return svg_clean


def strip_svg_text(svg_string: str,) -> str:
    """
    Remove all textual elements from an SVG string.

    The function tries to parse the SVG with BeautifulSoup (XML parser) and
    deletes the following tags if present: <text>, <tspan>, <title>, <desc>.
    If BeautifulSoup is unavailable or the SVG is malformed, a regex‐based
    fallback is used.

    Parameters
    ----------
    svg_string : str
        Raw SVG markup.

    Returns
    -------
    str
        SVG markup without any text elements.
    """
    # ------------------------------------------------------------------ #
    # 1) XML-based removal (preferred)                                   #
    # ------------------------------------------------------------------ #
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(svg_string, "xml")

        # Remove all text-related nodes
        for tag_name in ("text", "tspan", "title", "desc", "picture"):
            for node in soup.find_all(tag_name):
                node.decompose()

        # Convert back to string
        cleaned = str(soup)

        # BeautifulSoup adds an XML header we do not need
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("<?xml")
        )

        return cleaned

    except Exception:
        # Either bs4 is not installed or parsing failed – fall back to regex
        pass

    # ------------------------------------------------------------------ #
    # 2) Regex-based fallback                                            #
    # ------------------------------------------------------------------ #
    patterns = [
        r"<text\b[^>]*>.*?</text>",
        r"<tspan\b[^>]*>.*?</tspan>",
        r"<title\b[^>]*>.*?</title>",
        r"<desc\b[^>]*>.*?</desc>",
        r"<tref\b[^>]*>.*?</tref>",
        r"<textpath\b[^>]*>.*?</textpath>",
        r"<foreignobject\b[^>]*>.*?</foreignobject>",
        r"<picture\b[^>]*>.*?</picture>",
    ]

    cleaned = svg_string
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    return cleaned

def strip_inline_elements(s: str, inline_tags: list[list[str]]) -> str:
    for start_marker, end_marker in inline_tags:
        # Create a non-greedy regex pattern that matches everything between start_marker and end_marker
        pattern = re.escape(start_marker) + r'.*?' + re.escape(end_marker)
        s = re.sub(pattern, '', s, flags=re.DOTALL)
    return s
    

    


    
def find_unclosed_tags(svg_content):
    all_tags_pattern = r"<(\w+)"
    self_closing_pattern = r"<\w+[^>]*\/>"
    all_tags = re.findall(all_tags_pattern, svg_content)
    self_closing_matches = re.findall(self_closing_pattern, svg_content)
    self_closing_tags = []
    
    for match in self_closing_matches:
        tag = re.search(all_tags_pattern, match)
        if tag:
            self_closing_tags.append(tag.group(1))    
    unclosed_tags = []
    
    for tag in all_tags:
        if all_tags.count(tag) > self_closing_tags.count(tag) + svg_content.count('</' + tag + '>'):
            unclosed_tags.append(tag)
    unclosed_tags = list(dict.fromkeys(unclosed_tags))
    
    return unclosed_tags

def rgba_to_rgb_with_white_bg(rgba_img):
    """
    Converts an RGBA image to an RGB image with a white background.

    Parameters:
        rgba_img (np.array): Input image in RGBA format.

    Returns:
        np.array: RGB image with white background.
    """
    # Separate the color and alpha channels
    rgb_channels = rgba_img[:, :, :3]
    alpha_channel = rgba_img[:, :, 3] / 255.0

    # Create a white background image
    white_bg = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Blend the RGBA image onto the white background
    rgb_img = (rgb_channels * alpha_channel[:, :, np.newaxis] +
               white_bg * (1 - alpha_channel[:, :, np.newaxis])).astype(np.uint8)

    return rgb_img

def parse_length(length_str):
    """
    Parses an SVG length string (like "100px" or "50") and returns a float.
    """
    if length_str is None:
        return None
    # Use regex to extract the numeric part
    match = re.match(r"([0-9.]+)", length_str)
    if match:
        return float(match.group(1))
    return None

def get_svg_original_size(svg_string):
    """
    Parses an SVG string and returns its original width and height.
    Checks the 'width' and 'height' attributes first, then falls back to the viewBox if necessary.
    """
    try:
        root = ET.fromstring(svg_string)
        
        # Try to get width and height attributes
        width = parse_length(root.get("width"))
        height = parse_length(root.get("height"))
        
        # If width/height are not defined, try to extract from the viewBox attribute
        if width is None or height is None:
            viewbox = root.get("viewBox")
            if viewbox:
                parts = viewbox.split()
                if len(parts) == 4:
                    # The viewBox is defined as: min-x, min-y, width, height.
                    width = float(parts[2])
                    height = float(parts[3])
        
        return width, height
    except:
        return None, None

def random_color():
    """Generate a random hex color string."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def randomize_svg_colors(svg_string):
    """
    Parse the SVG, randomize colors for fill and stroke attributes,
    and return the modified SVG string.
    """
    root = ET.fromstring(svg_string)
    color_attribs = ['fill', 'stroke']

    def process_element(elem):
        # Update direct attributes
        for attrib in color_attribs:
            if attrib in elem.attrib:
                current_value = elem.attrib[attrib].strip().lower()
                # Only change if it's a color value (skip 'none')
                if current_value != "none":
                    elem.attrib[attrib] = random_color()

        # Process style attribute if present (e.g., style="fill:blue; stroke:black")
        if "style" in elem.attrib:
            style_props = elem.attrib["style"].split(";")
            new_style_props = []
            for prop in style_props:
                if prop.strip() == "":
                    continue
                try:
                    key, value = prop.split(":")
                except ValueError:
                    # Skip badly formatted style properties.
                    continue
                key = key.strip()
                value = value.strip()
                if key in color_attribs and value.lower() != "none":
                    value = random_color()
                new_style_props.append(f"{key}:{value}")
            elem.attrib["style"] = ";".join(new_style_props)

        # Recursively process child elements
        for child in elem:
            process_element(child)

    process_element(root)
    return ET.tostring(root, encoding='unicode')


def pil_rgba_to_rgb(img):
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    return background

def svg_to_np(svg_string, output_width=None, output_height=None, outline_only=False):
    # Render SVG to PNG bytes without background color to preserve alpha

    png_bytes = cairosvg.svg2png(
        bytestring=svg_string,
        output_width=output_width,
        output_height=output_height
    )

    # Load image with PIL
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode == "RGBA":
        # Convert RGBA to RGB with white background
        img = pil_rgba_to_rgb(img)
    
    # Convert to numpy array
    img = np.array(img)
    
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    return img

def pil_png_to_np(img):

    if img.mode == "RGBA":
        # Convert RGBA to RGB with white background
        img = pil_rgba_to_rgb(img)
    
    # Convert to numpy array
    img = np.array(img)
    
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    return img

def additive_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return noisy_image

def salt_pepper_noise(image, noise_level=0.1):
    noisy_image = np.copy(image)
    num_salt = np.ceil(noise_level * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 1
    num_pepper = np.ceil(noise_level * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0
    return noisy_image


def text_to_img(
    text: str,
    h: int,
    w: int,
    font_path: str = None,
    font_size: int = 24,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    start_y: int = 30,
    left_margin: int = 10,
    line_spacing: int = 5,
):
    """
    Render arbitrary (including Chinese) text into a (h, w, 3) BGR image.

    Unlike cv2.putText, this implementation uses Pillow so it supports
    Unicode characters out-of-the-box. Text is wrapped to stay inside the
    canvas bounds.

    Parameters
    ----------
    text : str
        Text to render.
    h, w : int
        Height / width of the resulting image.
    font_path : str, optional
        Path to a TrueType font. If None we try to use a sensible default.
        Make sure the chosen font contains the glyphs you need (e.g. a
        NotoSansCJK variant for Chinese).
    font_size : int
        Font size in pixels.
    text_color, bg_color : Tuple[int, int, int]
        BGR colours for text and background.
    start_y : int
        Initial vertical offset.
    left_margin : int
        Horizontal margin on both sides.
    line_spacing : int
        Extra pixels inserted between successive lines.

    Returns
    -------
    np.ndarray
        BGR uint8 image of shape (h, w, 3).
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont

    # 1. Prepare font ──────────
    if font_path is None:
        # Try a CJK‐capable font, then fall back to default.
        fallback_fonts = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # common Linux path
            "/System/Library/Fonts/STHeiti Medium.ttc",               # macOS
            "arial.ttf",
        ]
        for fp in fallback_fonts:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except (IOError, OSError):
                font = None
        if font is None:  # ultimate fallback
            font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)

    # 2. Create canvas ──────────
    pil_img = PILImage.new("RGB", (w, h), color=bg_color[::-1])  # Pillow uses RGB
    draw = ImageDraw.Draw(pil_img)

    max_text_width = w - 2 * left_margin

    # 3. Manual line wrapping (handles mixed Chinese / English) ──────────
    lines = []
    current_line = ""
    for char in text.replace("\r", ""):  # normalize newlines
        # Break on explicit newline
        if char == "\n":
            lines.append(current_line)
            current_line = ""
            continue

        test_line = current_line + char
        if draw.textlength(test_line, font=font) > max_text_width and current_line:
            # Current line is full, push to buffer
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    # 4. Render each line ──────────
    y = start_y
    for line in lines:
        if y + font_size > h:
            break  # Do not draw beyond bottom edge
        draw.text((left_margin, y), line, font=font, fill=text_color[::-1])  # convert BGR→RGB
        y += font_size + line_spacing

    # 5. Convert back to OpenCV (BGR) format ──────────
    img_np = np.array(pil_img)[:, :, ::-1].copy()  # RGB→BGR
    return img_np

def extract_svg(text: str) -> str:
    """
    Return the *last* complete SVG (from <svg …> to </svg>) found in ``text``.

    Parameters
    ----------
    text : str
        Text that may contain one or more SVG snippets.

    Returns
    -------
    str
        The last SVG string if one is found, otherwise an empty string.
    """
    pattern = r'<svg\b[^>]*>.*?</svg>'
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
    return matches[-1].group(0) if matches else ""


    
import matplotlib.pyplot as plt
def show_img_np(img, max_h=3, max_w=20, save="", cmap='gray'):
    """
    :param np_array: input image, one channel or 3 channel,
    :param save: if save image
    :param size:
    :return:
    """
    if len(img.shape) < 3:
        plt.rcParams['image.cmap'] = cmap
    plt.figure(figsize=(max_w, max_h), facecolor='w', edgecolor='k')
    plt.imshow(img)
    if save:
        cv2.imwrite(save, img)
    else:
        plt.show()
        

def gen_svg_pred_grid(img_gt, img_preds, alpha: float = 0.5, cell_size: int = 512):
    """
    Create a grid where every cell shows an equal-weight (0.5 / 0.5) overlay of
    the ground-truth image (`img_gt`) and one prediction from `img_preds`.

    Parameters
    ----------
    img_gt : np.ndarray
        Ground-truth image array shaped (H, W, 3).
    img_preds : List[np.ndarray]
        List of prediction images, each shaped (H, W, 3).
    alpha : float, optional
        Transparency weight for the prediction (default 0.5 ⇒ equal blend).
    cell_size : int, optional
        Size (in pixels) of each grid cell. Default: 512.

    Returns
    -------
    np.ndarray
        A canvas containing the blended images arranged in a square grid.
    """
    import math
    import numpy as np

    if len(img_preds) == 0:
        # Nothing to blend; just return the ground-truth image.
        return img_gt.copy()

    # Determine the square grid dimension (ceil of sqrt of number of preds).
    grid_size = max(1, int(math.ceil(math.sqrt(len(img_preds)))))

    # Initialize a white canvas to hold all blended cells.
    canvas = np.ones((grid_size * cell_size, grid_size * cell_size, 3),
                     dtype=np.uint8) * 255

    for idx, img_pred in enumerate(img_preds):
        # Blend ground truth and prediction with equal transparency.
        blended = ((1 - alpha) * img_gt + alpha * img_pred).clip(0, 255).astype(np.uint8)

        # Compute placement coordinates within the canvas.
        row, col = divmod(idx, grid_size)
        y0, y1 = row * cell_size, (row + 1) * cell_size
        x0, x1 = col * cell_size, (col + 1) * cell_size

        canvas[y0:y1, x0:x1] = blended

    return canvas

def visualize_rewards(
    img_gt,
    img_preds,
    rewards,
    *,
    alpha: float = 1.0,
    cell_px: int = 512,
    cmap_name: str = "nipy_spectral",
    show_img: bool = True,
):
    """
    Visualise rewards for up-to 1 GT + 16 predictions in a square grid.

    Changes compared to the previous version
    ----------------------------------------
    1. *All* reward keys are rendered – the earlier hard-coded cut–off has been
       removed and the vertical spacing is computed dynamically so everything
       fits.
    2. Metric texts are now drawn to the **left** of the image rather than
       above it, keeping the vertical space untouched.
    3. The prediction with the best score is enclosed in a blue box, the median
       in a green box, and the worst in a red box.
       
    Parameters
    ----------
    img_gt : np.ndarray
        Ground-truth image (H×W×3, uint8).
    img_preds : List[np.ndarray]
        List of prediction images (each H×W×3, uint8).
    rewards : List[Dict[str, float]]
        One rewards dictionary per prediction.
    alpha : float
        Blend weight for the prediction when overlaying on the GT.
    cell_px : int
        Unused for now – kept for API compatibility.
    cmap_name : str
        Matplotlib colour-map for reward → coloured background.
    show_img : bool
        If True, `plt.show()` is called before the NumPy frame is returned.

    Returns
    -------
    np.ndarray
        The visualisation as an RGB array (uint8).
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # --------------------------------------------------------------------- #
    # Quick sanity checks
    # --------------------------------------------------------------------- #
    assert len(img_preds) == len(rewards), "len(img_preds) must match len(rewards)"
    
    # Precompute main scores for each prediction using 'overall' (or 'image' if absent)
    main_scores = [float(r.get("overall", r.get("image", 0.0))) for r in rewards]
    sorted_indices = sorted(range(len(main_scores)), key=lambda i: main_scores[i])
    best_idx = sorted_indices[-1] if sorted_indices else None
    worst_idx = sorted_indices[0] if sorted_indices else None
    median_idx = sorted_indices[len(sorted_indices) // 2] if sorted_indices else None

    def _blend(a, b, t):
        return ((1.0 - t) * a + t * b).clip(0, 255).astype(np.uint8)

    # --------------------------------------------------------------------- #
    # Grid geometry
    # --------------------------------------------------------------------- #
    n_preds = len(img_preds)
    n_tiles = n_preds + 1  # +1 ⇒ ground-truth slot
    grid = int(math.ceil(math.sqrt(n_tiles)))  # square grid

    # A little wider than tall to make room for reward labels on the left
    fig, axes = plt.subplots(
        grid,
        grid,
        figsize=(grid * 3.6, grid * 3),  # wider
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.55, hspace=0.05)  # more horizontal spacing

    # Colour-map helpers
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(cmap_name)

    # --------------------------------------------------------------------- #
    # Ground-truth (row=0, col=0)
    # --------------------------------------------------------------------- #
    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title("GT", fontweight="bold", fontsize=8, pad=2)
    axes[0, 0].axis("off")
    for spine in axes[0, 0].spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(2)

    # --------------------------------------------------------------------- #
    # Predictions
    # --------------------------------------------------------------------- #
    for idx, (img_pred, rdict) in enumerate(zip(img_preds, rewards)):
        row, col = divmod(idx + 1, grid)  # +1 keeps GT at (0,0)
        ax = axes[row, col]

        ax.imshow(_blend(img_gt, img_pred, alpha))
        ax.axis("off")

        # Reorder keys so that primary metric ('overall' or 'image') is first
        keys = list(rdict.keys())
        for primary in ("overall", "image"):
            if primary in keys:
                keys.insert(0, keys.pop(keys.index(primary)))
                break  # only move one of them

        # Dynamic vertical spacing to ensure everything fits
        to_display = [k for k in keys if k != "error"]
        n_metrics = len(to_display)
        dy = 0.9 / max(1, n_metrics)  # reserve a comfortable margin at top/bottom
        y0 = 0.95  # start near the top inside the axes

        # Draw each metric on the left of the image
        for i, k in enumerate(to_display):
            val = float(rdict[k])
            colour = cmap(norm(val))
            ax.text(
                -0.02,                                # slightly outside on the left
                y0 - i * dy,
                f"{k}: {val:+.2f}",
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold" if k in ("overall", "image") else "normal",
                color="black" if norm(val) > 0.6 else "white",
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(
                    facecolor=colour,
                    edgecolor="none",
                    alpha=0.85,
                    pad=2,
                ),
                clip_on=False,
            )

        # Draw colored box around the best, median, and worst predictions.
        if idx == best_idx:
            rect_color = "blue"
        elif idx == median_idx:
            rect_color = "yellow"
        elif idx == worst_idx:
            rect_color = "red"
        else:
            rect_color = None

        if rect_color is not None:
            # Add a colored rectangle patch around the entire axes.
            rect = plt.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                fill=False,
                edgecolor=rect_color,
                linewidth=8
            )
            ax.add_patch(rect)

    # --------------------------------------------------------------------- #
    # Hide unused axes
    # --------------------------------------------------------------------- #
    for ax in axes.ravel()[n_tiles:]:
        ax.axis("off")

    # Global colour-bar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.03,
        pad=0.02,
        label="Reward  (-1 bad  |  1 good)",
    )
    cbar.ax.tick_params(labelsize=8)

    if show_img:
        plt.show()

    # --------------------------------------------------------------------- #
    # Convert figure → numpy
    # --------------------------------------------------------------------- #
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "tostring_rgb"):  # Matplotlib < 3.8
        buf = fig.canvas.tostring_rgb()
        img_np = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    else:  # Matplotlib ≥ 3.8
        buf = fig.canvas.buffer_rgba()
        img_np = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[..., :3]

    plt.close(fig)
    return img_np