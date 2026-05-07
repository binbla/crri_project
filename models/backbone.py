from mmdet.models.backbones.resnet import ResNet
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
import os
import sys
import hashlib
from collections import OrderedDict
from mmcv.cnn import build_conv_layer, build_norm_layer


@MODELS.register_module()
class CustomResNet(ResNet):
    """
    Custom ResNet that splits the forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_stage0(self, x):
        """

        Args:
            x (_type_): _description_
        """
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        return x

    def extract_stage1(self, x):
        """
        Run Stem + Layer 1.
        Output shape: [B, 256, H/4, W/4] (for ResNet50)
        """
        # 1. Stem (Image -> Stride 4, 64ch)
        # if self.deep_stem:
        #     x = self.stem(x)
        # else:
        #     x = self.conv1(x)
        #     x = self.norm1(x)
        #     x = self.relu(x)
        # x = self.maxpool(x)

        # 2. Layer 1 (Stride 4, 64ch -> 256ch if Bottleneck)
        layer1_name = self.res_layers[0]
        layer1 = getattr(self, layer1_name)
        x = layer1(x)

        return x

    def extract_stage2(self, x):
        """

        Args:
            x (_type_): _description_
        """
        layer2_name = self.res_layers[1]
        layer2 = getattr(self, layer2_name)
        x = layer2(x)
        return x

    def extract_stage3(self, x):
        """

        Args:
            x (_type_): _description_
        """
        layer3_name = self.res_layers[2]
        layer3 = getattr(self, layer3_name)
        x = layer3(x)
        return x

    def extract_stage4(self, x):
        """

        Args:
            x (_type_): _description_
        """
        layer4_name = self.res_layers[3]
        layer4 = getattr(self, layer4_name)
        x = layer4(x)
        return x

    def forward_from_stage0(self, x):
        """

        Args:
            x (_type_): _description_
        """
        outs = []
        for i, layer_name in enumerate(self.res_layers[0:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage1(self, x):
        """
        Continue forward pass from Layer 1 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.res_layers[1:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 1
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage2(self, x1, x2):
        """
        Continue forward pass from Layer 2 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)

        x = x2

        for i, layer_name in enumerate(self.res_layers[2:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 2
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage3(self, x1, x2, x3):
        """
        Continue forward pass from Layer 3 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)
        if 2 in self.out_indices:
            outs.append(x3)

        x = x3

        for i, layer_name in enumerate(self.res_layers[3:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 3
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage4(self, x1, x2, x3, x4):
        """
        Continue forward pass from Layer 4 to output.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)
        if 2 in self.out_indices:
            outs.append(x3)
        if 3 in self.out_indices:
            outs.append(x4)

        return tuple(outs)


'''
data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
'''
clip_path = os.path.join(os.path.dirname(__file__), 'CLIP')
if clip_path not in sys.path:
    sys.path.append(clip_path)

from clip import clip
from clip.model import Bottleneck, AttentionPool2d
'''
RN50	64	[3, 4, 6, 3]	32	1024
RN101	64	[3, 4, 23, 3]	32	512
RN50x4	80	[4, 6, 10, 6]	40	640
RN50x16	96	[6, 8, 18, 8]	48	768
RN50x64	128	[3, 15, 36, 10]	64	1024
'''


def _tensor_stats(t: torch.Tensor):
    t = t.detach().float().cpu()
    return {
        "shape": tuple(t.shape),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_mean": float(t.abs().mean()),
        "norm": float(t.norm()),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def _stats_str(name, stats):
    return (f"{name}: shape={stats['shape']}, "
            f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"abs_mean={stats['abs_mean']:.6f}, norm={stats['norm']:.6f}, "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}")


def _sha256_prefix(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@MODELS.register_module()
class CLIPResNet(BaseModule):
    MODEL_CONFIGS = {
        "RN50": {
            "width": 64,
            "layers": [3, 4, 6, 3],
            "heads": 32,
            "output_dim": 1024
        },
        "RN101": {
            "width": 64,
            "layers": [3, 4, 23, 3],
            "heads": 32,
            "output_dim": 512
        },
        "RN50x4": {
            "width": 80,
            "layers": [4, 6, 10, 6],
            "heads": 40,
            "output_dim": 640
        },
        "RN50x16": {
            "width": 96,
            "layers": [6, 8, 18, 8],
            "heads": 48,
            "output_dim": 768
        },
        "RN50x64": {
            "width": 128,
            "layers": [3, 15, 36, 10],
            "heads": 64,
            "output_dim": 1024
        },
    }

    def __init__(
            self,
            model_type="RN50",
            pretrained=None,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            layers=None,
            width=None,
            heads=None,
            output_dim=None,
            input_resolution=224,
            init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        if model_type in self.MODEL_CONFIGS:
            cfg = self.MODEL_CONFIGS[model_type]
            self.layers = layers or cfg['layers']
            self.width = width or cfg['width']
            self.heads = heads or cfg['heads']
            self.output_dim = output_dim or cfg['output_dim']
        else:
            raise ValueError(f"Unknown model_type: {model_type}. "
                             f"Available: {list(self.MODEL_CONFIGS.keys())}")

        self.model_type = model_type
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.input_resolution = input_resolution

        self.__ModifiedResNet_init__(layers=self.layers,
                                     output_dim=self.output_dim,
                                     heads=self.heads,
                                     input_resolution=self.input_resolution,
                                     width=self.width)

        self._freeze_stages()
        for param in self.attnpool.parameters():
            param.requires_grad = False

    def __init__old(
            self,
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            frozen_stages=-1,
            layers=None,
            width=None,
            heads=None,
            output_dim=None,
            input_resolution=224,
            init_cfg=None):

        if pretrained in self.MODEL_CONFIGS:
            cfg = self.MODEL_CONFIGS[pretrained]
            layers = layers or cfg['layers']
            width = width or cfg['width']
            heads = heads or cfg['heads']
            output_dim = output_dim or cfg['output_dim']

        assert layers is not None and output_dim is not None, "Must provide model config or a valid pretrained name"

        super().__init__(init_cfg=init_cfg)
        self.__ModifiedResNet_init__(layers=layers,
                                     output_dim=output_dim,
                                     heads=heads,
                                     input_resolution=input_resolution,
                                     width=width)

        self.pretrained = pretrained
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self._freeze_stages()
        for param in self.attnpool.parameters():
            param.requires_grad = False

    # ===== ModifiedResNet =====
    def __ModifiedResNet_init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    # ===== ModifiedResNet =====
    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for m in [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        if x.dtype != self.conv1.weight.dtype:
            x = x.to(self.conv1.weight.dtype)

        x = self.stem(x)

        outs = []
        # Layer 1
        x = self.layer1(x)
        if 0 in self.out_indices:
            outs.append(x)

        # Layer 2
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)

        # Layer 3
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)

        # Layer 4
        x = self.layer4(x)
        if 3 in self.out_indices:
            outs.append(x)

        # if 4 in self.out_indices:
        #     outs.append(self.attnpool(x))

        return tuple(outs)

    def extract_stage0(self, x):
        x = x.type(self.conv1.weight.dtype)
        # Stem (Image -> Stride 4, 64ch)
        x = self.stem(x)
        return x

    def extract_stage1(self, x):
        # x = x.type(self.conv1.weight.dtype)
        # Stem (Image -> Stride 4, 64ch)
        # x = self.stem(x)
        x = self.layer1(x)
        return x

    def extract_stage2(self, x):
        x = self.layer2(x)
        return x

    def extract_stage3(self, x):
        x = self.layer3(x)
        return x

    def extract_stage4(self, x):
        x = self.layer4(x)
        return x

    def forward_from_stage1(self, x):
        """
        Continue forward pass from Layer 1 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, layer_name in enumerate(layer_names[1:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 1
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage2(self, x1, x2):
        """
        Continue forward pass from Layer 2 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)

        x = x2

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, layer_name in enumerate(layer_names[2:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 2
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage3(self, x1, x2, x3):
        """
        Continue forward pass from Layer 3 to Layer 4.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)
        if 2 in self.out_indices:
            outs.append(x3)

        x = x3

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, layer_name in enumerate(layer_names[3:]):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_idx = i + 3
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward_from_stage4(self, x1, x2, x3, x4):
        """
        Continue forward pass from Layer 4 to output.
        Input x: Output of Layer 1 (potentially modified).
        """
        outs = []
        if 0 in self.out_indices:
            outs.append(x1)
        if 1 in self.out_indices:
            outs.append(x2)
        if 2 in self.out_indices:
            outs.append(x3)
        if 3 in self.out_indices:
            outs.append(x4)

        return tuple(outs)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        logger.info("=" * 80)
        logger.info(f"[CLIPResNet] init_weights() called")
        logger.info(f"[CLIPResNet] model_type={self.model_type}, pretrained={self.pretrained}")

        # ------------------------------------------------------------------
        # 1) resolve ckpt path
        # ------------------------------------------------------------------
        if not isinstance(self.pretrained, str):
            raise RuntimeError(
                f"[CLIPResNet] Expected local pretrained path string, got: {self.pretrained}")

        ckpt_path = self.pretrained
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[CLIPResNet] pretrained file not found: {ckpt_path}")

        file_size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        sha_prefix = _sha256_prefix(ckpt_path)
        logger.info(f"[CLIPResNet] checkpoint path={ckpt_path}, "
                    f"size={file_size_mb:.2f} MB, sha256[:12]={sha_prefix}")

        # ------------------------------------------------------------------
        # 2) load raw state_dict
        # ------------------------------------------------------------------
        state_dict = None
        load_format = None

        try:
            jit_model = torch.jit.load(ckpt_path, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
            load_format = "torchscript"
            logger.info("[CLIPResNet] Loaded checkpoint with torch.jit.load()")
        except Exception as e:
            logger.warning(f"[CLIPResNet] torch.jit.load failed: {repr(e)}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                    load_format = "state_dict"
                elif "model" in ckpt:
                    state_dict = ckpt["model"]
                    load_format = "model"
                else:
                    state_dict = ckpt
                    load_format = "raw_dict"
            else:
                raise RuntimeError(f"[CLIPResNet] Unsupported checkpoint object type: {type(ckpt)}")

            logger.info(f"[CLIPResNet] Loaded checkpoint with torch.load(), format={load_format}")

        if not isinstance(state_dict, dict):
            raise RuntimeError("[CLIPResNet] state_dict is not a dict after loading")

        logger.info(f"[CLIPResNet] raw checkpoint keys: {len(state_dict)}")
        sample_keys = list(state_dict.keys())[:20]
        logger.info(f"[CLIPResNet] raw checkpoint sample keys: {sample_keys}")

        # ------------------------------------------------------------------
        # 3) filter / rename keys
        # ------------------------------------------------------------------
        model_state = self.state_dict()
        mapped_state = OrderedDict()

        filtered_text = []
        filtered_attnpool = []
        unexpected_keys = []
        shape_mismatch = []

        for k, v in state_dict.items():
            orig_k = k

            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("model."):
                k = k[len("model."):]

            if (k.startswith("transformer.") or k.startswith("token_embedding")
                    or k.startswith("ln_final") or k.startswith("text_projection")
                    or k == "positional_embedding" or k == "logit_scale"):
                filtered_text.append(orig_k)
                continue

            if k.startswith("visual."):
                k = k[len("visual."):]

            if k.startswith("attnpool."):
                filtered_attnpool.append(orig_k)
                continue

            if k.startswith("res2."):
                k = "layer1." + k[len("res2."):]
            elif k.startswith("res3."):
                k = "layer2." + k[len("res3."):]
            elif k.startswith("res4."):
                k = "layer3." + k[len("res4."):]
            elif k.startswith("res5."):
                k = "layer4." + k[len("res5."):]

            if k not in model_state:
                unexpected_keys.append((orig_k, k, tuple(v.shape)))
                continue

            if tuple(v.shape) != tuple(model_state[k].shape):
                shape_mismatch.append((orig_k, k, tuple(v.shape), tuple(model_state[k].shape)))
                continue

            mapped_state[k] = v

        logger.info(f"[CLIPResNet] filtered text keys: {len(filtered_text)}")
        logger.info(f"[CLIPResNet] filtered attnpool keys: {len(filtered_attnpool)}")
        logger.info(f"[CLIPResNet] matched loadable keys: {len(mapped_state)}")
        logger.info(f"[CLIPResNet] unexpected keys after mapping: {len(unexpected_keys)}")
        logger.info(f"[CLIPResNet] shape mismatch keys: {len(shape_mismatch)}")

        if unexpected_keys:
            logger.info(f"[CLIPResNet] unexpected sample: {unexpected_keys[:10]}")
        if shape_mismatch:
            logger.warning(f"[CLIPResNet] shape mismatch sample: {shape_mismatch[:10]}")

        total_model_keys = len(model_state)
        ratio = len(mapped_state) / max(total_model_keys, 1)
        logger.info(
            f"[CLIPResNet] loadable ratio = {len(mapped_state)}/{total_model_keys} = {ratio:.4f}")
        if len(mapped_state) == 0:
            raise RuntimeError(
                "[CLIPResNet] No keys matched. The model is effectively random init.")
        if ratio < 0.3:
            logger.warning(
                "[CLIPResNet] Very low matched ratio. Check model_type / architecture / naming.")

        # ------------------------------------------------------------------
        # 4) probe tensors: before load
        # ------------------------------------------------------------------
        preferred_probe_keys = [
            "conv1.weight",
            "bn1.weight",
            "bn1.bias",
            "layer1.0.conv1.weight",
            "layer1.0.bn1.weight",
            "layer2.0.conv1.weight",
            "layer3.0.conv1.weight",
            "layer4.0.conv1.weight",
        ]
        probe_keys = [k for k in preferred_probe_keys if k in mapped_state and k in model_state]

        if len(probe_keys) < 5:
            for k in mapped_state.keys():
                if k not in probe_keys:
                    probe_keys.append(k)
                if len(probe_keys) >= 8:
                    break

        logger.info(f"[CLIPResNet] probe keys: {probe_keys}")

        before_stats = {}
        ckpt_stats = {}
        for k in probe_keys:
            before_stats[k] = _tensor_stats(model_state[k])
            ckpt_stats[k] = _tensor_stats(mapped_state[k])
            logger.info("[BEFORE] " + _stats_str(k, before_stats[k]))
            logger.info("[CKPT  ] " + _stats_str(k, ckpt_stats[k]))

        # ------------------------------------------------------------------
        # 5) load
        # ------------------------------------------------------------------
        load_result = self.load_state_dict(mapped_state, strict=False)

        self.float()

        # ------------------------------------------------------------------
        # 6) verify after load
        # ------------------------------------------------------------------
        model_state_after = self.state_dict()
        verified_ok = 0

        for k in probe_keys:
            after = model_state_after[k].detach().cpu().float()
            ref = mapped_state[k].detach().cpu().float()

            after_stats = _tensor_stats(after)
            logger.info("[AFTER ] " + _stats_str(k, after_stats))

            diff = (after - ref).abs()
            max_abs_diff = float(diff.max())
            mean_abs_diff = float(diff.mean())
            exact_equal = bool(torch.equal(after, ref))
            allclose = bool(torch.allclose(after, ref, atol=0, rtol=0))

            logger.info(f"[VERIFY] {k}: exact_equal={exact_equal}, allclose={allclose}, "
                        f"max_abs_diff={max_abs_diff:.12f}, mean_abs_diff={mean_abs_diff:.12f}")

            if allclose:
                verified_ok += 1

        logger.info(f"[CLIPResNet] verified probe tensors: {verified_ok}/{len(probe_keys)}")

        # ------------------------------------------------------------------
        # 7) report missing keys
        # ------------------------------------------------------------------
        ignored_missing_prefixes = ("attnpool.", )
        meaningful_missing = [
            k for k in load_result.missing_keys
            if not k.startswith(ignored_missing_prefixes) and "num_batches_tracked" not in k
        ]

        logger.info(f"[CLIPResNet] missing_keys(total)={len(load_result.missing_keys)}")
        logger.info(f"[CLIPResNet] unexpected_keys(total)={len(load_result.unexpected_keys)}")

        if meaningful_missing:
            logger.warning(f"[CLIPResNet] meaningful missing keys ({len(meaningful_missing)}): "
                           f"{meaningful_missing[:30]}")

        if verified_ok == 0:
            raise RuntimeError(
                "[CLIPResNet] Load finished but none of the probe tensors matched checkpoint values."
            )

        # sign for debug
        self._clip_init_done = True
        self._clip_init_ckpt = ckpt_path
        self._clip_init_sha256_prefix = sha_prefix

        logger.info(f"[CLIPResNet] SUCCESS: weights loaded from {ckpt_path} "
                    f"(sha256[:12]={sha_prefix})")
        logger.info("=" * 80)

