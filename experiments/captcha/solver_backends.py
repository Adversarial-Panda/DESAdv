"""Model backends for captcha_solver.py.

Every backend maps a list of tile images to one of the 10 CAPTCHA classes per tile
(closed-set decision, like the earlier CLIP baseline). No labels are read here.
"""
import os
import re
from pathlib import Path

# All model weights live in ./weights (see README.md; download_weights.py recreates it).
# Must be set before huggingface_hub / transformers are imported.
WEIGHTS_DIR = Path(os.environ.get('CAPTCHA_WEIGHTS', Path(__file__).resolve().parent / 'weights'))
os.environ.setdefault('HF_HUB_CACHE', str(WEIGHTS_DIR / 'huggingface'))


def ultralytics_weight(name):
    """Bare Ultralytics weight names resolve to weights/ultralytics/ (downloaded there if missing)."""
    return name if os.path.exists(name) else str(WEIGHTS_DIR / 'ultralytics' / Path(name).name)


# Xet transfers stall or fail with "CAS Client Error" on this network; plain HTTP downloads work.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
# Reduces fragmentation-related out-of-memory errors on 8 GB GPUs (read when CUDA initialises).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

CLASSES = ['goldfish', 'tiger', 'zebra', 'school bus', 'banana',
           'ant', 'pizza', 'bee', 'traffic light', 'elephant']

# Short names usable with --model. Sizes chosen to fit an 8 GB GPU;
# "4bit" presets are quantized on load with bitsandbytes.
PRESETS = {
    'clip':            {'backend': 'clip', 'model': 'openai/clip-vit-base-patch32'},
    'yolov8':          {'backend': 'yolo', 'model': 'yolov8m-cls.pt'},
    'yolov8-cam':      {'backend': 'yolo-cam', 'model': 'yolov8m-cls.pt'},
    'yolo-world':      {'backend': 'yolo-world', 'model': 'yolov8x-worldv2.pt'},
    'owlv2':           {'backend': 'owlv2', 'model': 'google/owlv2-base-patch16-ensemble'},
    'qwen2-vl':        {'backend': 'vlm', 'model': 'Qwen/Qwen2-VL-2B-Instruct'},
    'qwen2-vl-7b':     {'backend': 'vlm', 'model': 'Qwen/Qwen2-VL-7B-Instruct', 'load_4bit': True},
    'qwen2.5-vl':      {'backend': 'vlm', 'model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'load_4bit': True},
    # Same weights as qwen2.5-vl, given the whole CAPTCHA and asked for bounding boxes (grounding).
    'qwen2.5-vl-ground': {'backend': 'vlm-ground', 'model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'load_4bit': True},
    # LLaVA-1.5 returns one box per question: ask, verify the crop, grey it out, ask again.
    'llava-ground':    {'backend': 'vlm-ground', 'model': 'llava-hf/llava-1.5-7b-hf', 'load_4bit': True},
    'qwen2.5-vl-7b':   {'backend': 'vlm', 'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'load_4bit': True},
    'llava':           {'backend': 'vlm', 'model': 'llava-hf/llava-1.5-7b-hf', 'load_4bit': True},
    'llava-onevision': {'backend': 'vlm', 'model': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf'},
    'internvl':        {'backend': 'vlm', 'model': 'OpenGVLab/InternVL3-2B-hf'},
    'internvl-8b':     {'backend': 'vlm', 'model': 'OpenGVLab/InternVL3-8B-hf', 'load_4bit': True},
}


def resolve(name, load_4bit=None):
    """Return (preset name used for output folders, backend, model id, load_4bit)."""
    by_model = {}
    for key, p in PRESETS.items():
        by_model.setdefault(p['model'], key)  # a model id maps to its first (default) preset
    if name in PRESETS or name in by_model:
        short = name if name in PRESETS else by_model[name]
        preset = PRESETS[short]
    else:
        preset = {'backend': 'yolo' if name.endswith('.pt') else 'vlm', 'model': name}
        short = re.sub(r'\.pt$', '', name.rstrip('/').split('/')[-1])
    quantize = preset.get('load_4bit', False) if load_4bit is None else load_4bit
    return short, preset['backend'], preset['model'], quantize


class YoloBackend:
    """ImageNet-pretrained YOLOv8-cls restricted to the CAPTCHA classes.
    COCO detection weights lack goldfish, tiger, ant and bee, so -cls weights are used."""
    IMAGENET_IDS = {'goldfish': [1], 'tiger': [292], 'zebra': [340], 'school bus': [779],
                    'banana': [954], 'ant': [310], 'pizza': [963], 'bee': [309],
                    'traffic light': [920], 'elephant': [101, 385, 386]}

    def __init__(self, model_id, device, imgsz=224, **_):
        from PIL import Image
        from ultralytics import YOLO
        self.device = 0 if device == 'cuda' else 'cpu'  # ultralytics expects a GPU index
        self.imgsz = imgsz
        self.model = YOLO(ultralytics_weight(model_id), task='classify')
        self.names = self.model.names
        if len(self.names) != 1000:
            raise SystemExit(f'{model_id} is not an ImageNet-1k classifier ({len(self.names)} classes).')
        print('Class mapping:', {c: [self.names[i] for i in ids]
                                 for c, ids in self.IMAGENET_IDS.items()}, flush=True)
        self.predict([Image.new('RGB', (imgsz, imgsz))])  # warm-up so round 1 timing is fair

    def scores(self, images, batch_size=256):
        """[N, 11]: probability mass of each CAPTCHA class, plus all other ImageNet classes ("no object")."""
        import torch
        out = []
        for start in range(0, len(images), batch_size):
            results = self.model.predict(images[start:start + batch_size], imgsz=self.imgsz,
                                         device=self.device, verbose=False)
            probs = torch.stack([r.probs.data.float().cpu() for r in results])
            classes = torch.stack([probs[:, ids].sum(dim=1) for ids in self.IMAGENET_IDS.values()], dim=1)
            out.append(torch.cat([classes, 1 - classes.sum(dim=1, keepdim=True)], dim=1))
        return torch.cat(out)

    def locate(self, image, target, threshold=0.5):
        """Area mode: same sliding windows as CLIP, scored by this ImageNet classifier."""
        return locate_with_windows(self.scores, image, target, threshold=threshold)

    def predict(self, images):
        import torch
        results = self.model.predict(images, imgsz=self.imgsz, device=self.device, verbose=False)
        probs = torch.stack([r.probs.data.float().cpu() for r in results])
        scores = torch.stack([probs[:, ids].sum(dim=1) for ids in self.IMAGENET_IDS.values()], dim=1)
        predictions = [CLASSES[i] for i in scores.argmax(dim=1).tolist()]
        return predictions, {'class_scores': scores.tolist(),
                             'imagenet_top1': [self.names[i] for i in probs.argmax(dim=1).tolist()]}


class YoloCamBackend(YoloBackend):
    """YOLOv8-cls given the whole CAPTCHA as one image. Its head is 1x1 conv -> global average pool ->
    linear, so applying the linear layer at every feature-map location gives exact per-location class
    scores (class activation map; equal to Grad-CAM for this head). Dots go on peaks of the target map."""
    full_resolution = True   # area canvas at its natural 600 px
    whole_image = True       # grid mode also uses the whole grid image, not per-tile classification

    def __init__(self, model_id, device, cam_width=1152, threshold=0.5, **options):
        super().__init__(model_id, device, **options)
        self.net = self.model.model.to('cuda' if device == 'cuda' else 'cpu').eval()
        self.head = self.net.model[-1]
        self.cam_width = cam_width   # objects end up near the 224 px scale the classifier was trained at
        self.threshold = threshold

    def class_maps(self, image):
        """[10, h, w] per-location probability of each CAPTCHA class (ImageNet ids summed)."""
        import numpy as np
        import torch
        from PIL import Image
        width = self.cam_width
        height = max(32, round(image.height * width / image.width / 32) * 32)
        pixels = np.asarray(image.convert('RGB').resize((width, height), Image.BILINEAR), np.float32) / 255
        device = next(self.net.parameters()).device
        x = torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device)
        captured = {}
        hook = self.head.register_forward_hook(lambda m, inputs, output: captured.update(x=inputs[0]))
        with torch.inference_mode():
            self.net(x)
            hook.remove()
            features = self.head.conv(captured['x'])                       # [1, 1280, h, w]
            weight, bias = self.head.linear.weight, self.head.linear.bias  # [1000, 1280], [1000]
            logits = torch.einsum('kc,bchw->bkhw', weight, features) + bias[None, :, None, None]
            probs = logits.float().softmax(dim=1)[0]                        # softmax per location
            maps = torch.stack([probs[ids].sum(dim=0) for ids in self.IMAGENET_IDS.values()])
        return maps.cpu()

    def locate(self, image, target):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        maps = self.class_maps(image)
        heat = torch.nn.functional.interpolate(maps[None], size=(image.height, image.width),
                                               mode='bilinear', align_corners=False)[0]
        target_map = heat[CLASSES.index(target)].numpy()
        # Peaks: local maxima over a window about the size of one pasted image, above the threshold.
        k = max(3, int(0.12 * image.width) | 1)
        local_max = cv2.dilate(target_map, np.ones((k, k), np.uint8))
        ys, xs = np.nonzero((target_map >= local_max - 1e-6) & (target_map >= self.threshold))
        peaks = sorted(((float(target_map[y, x]), int(x), int(y)) for y, x in zip(ys, xs)), reverse=True)
        points = []
        for score, x, y in peaks:
            if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 > 0.08 * image.width for px, py in points):
                points.append((x, y))
        overlay = cv2.applyColorMap((target_map * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        heatmap = Image.blend(image.convert('RGB'), Image.fromarray(overlay.copy()), 0.45)
        return points, {'peaks': [p for p in peaks if (p[1], p[2]) in points], 'threshold': self.threshold,
                        'cam_input': [self.cam_width, int(maps.shape[1] * 32)], 'cam_grid': list(maps.shape[1:]),
                        'kept_boxes': [], 'heatmap': heatmap}


def locate_with_windows(score_fn, image, target, sizes=(0.14, 0.18, 0.22, 0.26), stride=0.035, threshold=0.5):
    """Sliding-window localisation shared by classifier backends.
    score_fn(crops) -> tensor [N, K]; columns 0-9 follow CLASSES, extra columns mean "no object".
    A window is a target when the target column is the largest and at least `threshold`.
    Returns (click points in image pixels, details)."""
    width, height = image.size
    boxes = []
    for fraction in sizes:
        side, step = round(fraction * width), max(4, round(stride * width))
        for y in range(0, height - side + 1, step):
            for x in range(0, width - side + 1, step):
                boxes.append((x, y, x + side, y + side))
    probs = score_fn([image.crop(box) for box in boxes])
    t = CLASSES.index(target)
    positive = [(probs[i, t].item(), boxes[i]) for i in range(len(boxes))
                if probs[i].argmax().item() == t and probs[i, t].item() >= threshold]

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    # Non-maximum suppression, then pull each point to the score-weighted centre of its cluster,
    # since a single window centre can sit on the edge of an object.
    points, kept = [], []
    for score, box in sorted(positive, reverse=True):
        if any(iou(box, k) > 0.2 for k in kept):
            continue
        kept.append(box)
        cluster = [(s, b) for s, b in positive if iou(box, b) > 0.3]
        total = sum(s for s, _ in cluster)
        cx = sum(s * (b[0] + b[2]) / 2 for s, b in cluster) / total
        cy = sum(s * (b[1] + b[3]) / 2 for s, b in cluster) / total
        if all(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > 0.08 * width for px, py in points):
            points.append((round(cx, 1), round(cy, 1)))
    return points, {'windows': len(boxes), 'positive_windows': len(positive),
                    'kept_boxes': kept, 'threshold': threshold}


class YoloWorldBackend:
    """YOLOv8-World open-vocabulary detector, prompted with the 10 class names (COCO YOLOv8 lacks
    goldfish, tiger, ant and bee). It sees the whole CAPTCHA image at once and returns boxes; the
    solver clicks the centre of every box labelled as the target."""

    # Area canvases are shown scaled down (600 -> 404 px); detection needs the full-resolution picture.
    full_resolution = True

    def __init__(self, model_id, device, conf=0.05, detect_size=960, **_):
        # conf 0.05 / 960 px input: best solve rate in offline tests (higher conf mainly misses targets).
        import ultralytics.nn.text_model as text_model
        from ultralytics import YOLOWorld
        self.device = 0 if device == 'cuda' else 'cpu'
        self.conf = conf
        self.imgsz = detect_size
        self.model = YOLOWorld(ultralytics_weight(model_id))
        # Its CLIP text encoder (ViT-B-32.pt) goes to weights/clip, whatever the working directory.
        text_model.WEIGHTS_DIR = WEIGHTS_DIR
        self.model.set_classes(CLASSES)

    def locate(self, image, target):
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        detections = [{'label': result.names[int(b.cls)], 'conf': round(float(b.conf), 3),
                       'box': [round(v, 1) for v in b.xyxy[0].tolist()]} for b in result.boxes]
        points, kept = [], []
        for d in sorted(detections, key=lambda d: -d['conf']):
            if d['label'] != target:
                continue
            x0, y0, x1, y1 = d['box']
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # Two dots on one object can toggle each other off in the app; keep the most confident.
            if all(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > 0.08 * image.width for px, py in points):
                points.append((round(cx, 1), round(cy, 1)))
                kept.append(d['box'])
        return points, {'detections': detections, 'kept_boxes': kept, 'conf_threshold': self.conf}


class Owlv2Backend:
    """OWLv2 open-vocabulary detector: the whole CAPTCHA plus one text query per class. A box counts for
    the target when the target query scores highest for it; the solver clicks each such box's centre."""
    full_resolution = True
    whole_image = True

    def __init__(self, model_id, device, threshold=0.2, **_):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.device = device
        self.threshold = threshold
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.queries = [f'a photo of a {c}' for c in CLASSES]
        with torch.inference_mode():
            self.text_inputs = self.processor(text=[self.queries], return_tensors='pt').to(device)

    def locate(self, image, target):
        import torch
        from torchvision.ops import nms
        image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, **self.text_inputs)
        # OWLv2 pads the image to a square (bottom/right) before resizing, so boxes are in that square.
        side = max(image.size)
        result = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=torch.tensor([[side, side]]))[0]
        t = CLASSES.index(target)
        keep = result['labels'] == t
        boxes, scores = result['boxes'][keep], result['scores'][keep]
        boxes = boxes[nms(boxes, scores, 0.3)] if len(boxes) else boxes
        points, kept = [], []
        for x1, y1, x2, y2 in boxes.tolist():
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if all(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > 0.08 * image.width for px, py in points):
                points.append((round(cx, 1), round(cy, 1)))
                kept.append([round(v, 1) for v in (x1, y1, x2, y2)])
        detections = [{'label': CLASSES[l], 'score': round(s, 3), 'box': [round(v, 1) for v in b]}
                      for l, s, b in zip(result['labels'].tolist(), result['scores'].tolist(), result['boxes'].tolist())]
        return points, {'detections': detections, 'kept_boxes': kept, 'threshold': self.threshold}


def _embedding(output):
    """transformers 5 returns an output object (projected embedding in pooler_output); 4.x a tensor."""
    features = output if hasattr(output, 'norm') else output.pooler_output
    return features / features.norm(dim=-1, keepdim=True)


class ClipBackend:
    """CLIP zero-shot classification over 'a photo of <class>' prompts (same as captcha_clip_solver.py).
    An extra background prompt lets area mode reject windows that show no object."""
    BACKGROUND_PROMPT = 'a blurry, noisy, colorful abstract background'

    def __init__(self, model_id, device, **_):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        prompts = [f'a photo of {c}' for c in CLASSES] + [self.BACKGROUND_PROMPT]
        with torch.inference_mode():
            text = self.processor(text=prompts, return_tensors='pt', padding=True).to(device)
            self.text = _embedding(self.model.get_text_features(**text))

    def probs(self, images, batch_size=256):
        """Softmax over the 10 classes plus background; shape [len(images), 11]."""
        import torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                inputs = self.processor(images=images[start:start + batch_size], return_tensors='pt').to(self.device)
                features = _embedding(self.model.get_image_features(**inputs))
                logits = self.model.logit_scale.exp() * features @ self.text.T
                out.append(logits.float().softmax(dim=-1).cpu())
        return torch.cat(out)

    def predict(self, images):
        import torch
        with torch.inference_mode():
            # Grid tiles always show an object: decide among the 10 classes only, as the original CLIP solver.
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            features = _embedding(self.model.get_image_features(**inputs))
            logits = (self.model.logit_scale.exp() * features @ self.text[:len(CLASSES)].T).float().cpu()
        return [CLASSES[i] for i in logits.argmax(dim=1).tolist()], {'logits': logits.tolist()}

    def locate(self, image, target, threshold=0.5):
        """Area mode: sliding windows scored by CLIP (10 classes + background prompt)."""
        return locate_with_windows(self.probs, image, target, threshold=threshold)

# Answers that do not name a class exactly but clearly mean one.
SYNONYMS = {'fish': 'goldfish', 'bus': 'school bus', 'traffic signal': 'traffic light',
            'stoplight': 'traffic light', 'stop light': 'traffic light', 'ants': 'ant',
            'bees': 'bee', 'honeybee': 'bee', 'wasp': 'bee', 'bananas': 'banana',
            'elephants': 'elephant', 'zebras': 'zebra', 'tigers': 'tiger'}


def parse_answer(text):
    """Map a free-text VLM answer to a class; earliest mention wins, else 'unknown'."""
    text = text.lower()
    hits = []
    for name, label in [(c, c) for c in CLASSES] + list(SYNONYMS.items()):
        match = re.search(rf'\b{re.escape(name)}\b', text)
        if match:
            hits.append((match.start(), -len(name), label))
    return min(hits)[2] if hits else 'unknown'


def parse_boxes(text):
    """Qwen2.5-VL grounding answer -> list of [x1, y1, x2, y2] (JSON if possible, else number quadruples)."""
    import json
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            boxes = [item.get('bbox_2d') for item in items if isinstance(item, dict)]
            return [[float(v) for v in b] for b in boxes if isinstance(b, list) and len(b) == 4]
        except (ValueError, TypeError, AttributeError):
            pass
    return [[float(v) for v in m] for m in re.findall(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)]


class VLMBackend:
    """Any Hugging Face image-text-to-text chat model (Qwen2-VL, Qwen2.5-VL, LLaVA, InternVL-hf).
    Each tile is asked to pick exactly one class from the fixed list; greedy decoding."""
    PROMPT = ('What is the main object in this image? Answer with exactly one option from '
              'this list and nothing else: ' + ', '.join(CLASSES) + '.')

    GROUNDING_PROMPT = ('Locate every {target} in the image and output the bbox coordinates '
                        'in JSON format.')

    def __init__(self, model_id, device, load_4bit=False, image_size=448, batch_size=9, grounding=False, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if device != 'cuda' and load_4bit:
            raise SystemExit('4-bit loading needs a CUDA GPU; pass --no-4bit to run on CPU (very slow).')
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        kwargs = {'dtype': dtype, 'device_map': 'auto' if device == 'cuda' else None}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = 'left'  # required for batched generation
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
        self.image_size = image_size
        self.batch_size = batch_size
        # Grounding: the whole CAPTCHA (area canvas at natural size, or the full grid) in one request.
        self.whole_image = self.full_resolution = grounding
        if grounding and self.model.config.model_type not in ('qwen2_5_vl', 'llava'):
            raise SystemExit('Grounding mode is implemented for Qwen2.5-VL and LLaVA-1.5 only.')
        if device == 'cuda':
            print(f'GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.1f} GiB', flush=True)

    def _generate(self, batch):
        import torch
        # Tiles are small browser screenshots; give every model the same upscaled input.
        conversations = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': img.resize((self.image_size, self.image_size))},
            {'type': 'text', 'text': self.PROMPT}]}] for img in batch]
        inputs = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors='pt', processor_kwargs={'padding': True}
        ).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        return self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
                                           skip_special_tokens=True)

    def predict(self, images):
        import torch
        answers = []
        start = 0
        while start < len(images):
            batch = images[start:start + self.batch_size]
            try:
                answers += self._generate(batch)
            except torch.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                # Larger models (e.g. LLaVA-1.5-7B) cannot fit 9 tiles at once on 8 GB; keep the smaller size.
                self.batch_size = max(1, self.batch_size // 2)
                print(f'GPU out of memory; retrying with --batch-size {self.batch_size}', flush=True)
                continue
            start += len(batch)
        answers = [a.strip() for a in answers]
        return [parse_answer(a) for a in answers], {'raw_answers': answers}

    LLAVA_BOX_PROMPT = 'Please provide the bounding box coordinate of the region this sentence describes: {target}.'
    LLAVA_CHECK_PROMPT = 'Is there a {target} in this image? Answer yes or no.'

    def _ask(self, image, prompt, max_new_tokens):
        import torch
        conversation = [[{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                      {'type': 'text', 'text': prompt}]}]]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors='pt', processor_kwargs={'padding': True}).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()

    @staticmethod
    def _square(image, fill=(127, 127, 127)):
        """Pad to a square so LLaVA's 336 px centre crop keeps the whole image; returns (square, x0, y0)."""
        from PIL import Image
        side = max(image.size)
        square = Image.new('RGB', (side, side), fill)
        x0, y0 = (side - image.width) // 2, (side - image.height) // 2
        square.paste(image, (x0, y0))
        return square, x0, y0

    def _llava_locate(self, image, target, max_objects=4):
        """LLaVA-1.5 answers one normalised box per question. Repeat on the whole image: ask for a box,
        verify the crop with a yes/no question, grey the region out, ask again; stop at the first 'no'."""
        from PIL import ImageDraw
        work = image.convert('RGB').copy()
        points, kept, answers = [], [], []
        for _ in range(max_objects):
            square, x0, y0 = self._square(work)
            side = square.width
            answer = self._ask(square, self.LLAVA_BOX_PROMPT.format(target=target), 40)
            numbers = re.findall(r'\d*\.\d+|\d+', answer)
            if len(numbers) < 4:
                answers.append(answer)
                break
            nx1, ny1, nx2, ny2 = (float(v) for v in numbers[:4])
            x1, y1 = max(0, nx1 * side - x0), max(0, ny1 * side - y0)
            x2, y2 = min(image.width, nx2 * side - x0), min(image.height, ny2 * side - y0)
            if x2 - x1 < 8 or y2 - y1 < 8:
                answers.append(answer)
                break
            crop, _, _ = self._square(image.crop((x1, y1, x2, y2)))
            check = self._ask(crop, self.LLAVA_CHECK_PROMPT.format(target=target), 3)
            answers.append(f'{answer} -> {check}')
            if not check.lower().startswith('yes'):
                break
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if all(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > 0.08 * image.width for px, py in points):
                points.append((round(cx, 1), round(cy, 1)))
                kept.append([round(v, 1) for v in (x1, y1, x2, y2)])
            ImageDraw.Draw(work).rectangle((x1, y1, x2, y2), fill=(127, 127, 127))  # hide it for the next question
        return points, {'raw_answer': ' | '.join(answers), 'kept_boxes': kept}

    def locate(self, image, target, upscale=2):
        """Grounding: ask for boxes of the target on the whole image; click each box centre.
        Qwen2.5-VL: one request; the image is upscaled 2x first (offline: 35% -> 52% solved)."""
        if not self.whole_image:
            raise RuntimeError('Per-tile VLM presets cannot solve area challenges; use the -ground preset '
                               '(qwen2.5-vl-ground, llava-ground).')
        if self.model.config.model_type == 'llava':
            return self._llava_locate(image, target)
        import torch
        from PIL import Image
        shown = image.convert('RGB').resize((image.width * upscale, image.height * upscale), Image.BICUBIC)
        conversation = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': shown},
            {'type': 'text', 'text': self.GROUNDING_PROMPT.format(target=target)}]}]]
        inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors='pt', processor_kwargs={'padding': True}).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        # Boxes are in pixels of the image as the model saw it (resized to multiples of 28); map them back.
        _, grid_h, grid_w = inputs['image_grid_thw'][0].tolist()
        sx, sy = image.width / (grid_w * 14), image.height / (grid_h * 14)
        boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in parse_boxes(answer)]
        points, kept = [], []
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # Two dots on one object can toggle each other off in the app; keep the first.
            if all(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > 0.08 * image.width for px, py in points):
                points.append((round(cx, 1), round(cy, 1)))
                kept.append([round(v, 1) for v in (x1, y1, x2, y2)])
        return points, {'raw_answer': answer, 'kept_boxes': kept, 'model_input_size': [grid_w * 14, grid_h * 14]}


def load_backend(backend, model_id, device, **options):
    if backend == 'vlm-ground':
        return VLMBackend(model_id, device, grounding=True, **options)
    cls = {'yolo': YoloBackend, 'yolo-cam': YoloCamBackend, 'clip': ClipBackend,
           'yolo-world': YoloWorldBackend, 'owlv2': Owlv2Backend}.get(backend, VLMBackend)
    return cls(model_id, device, **options)
