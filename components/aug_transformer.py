from typing import Optional, Tuple, Union
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
import random
import math
import torch
from mmdet.registry import TRANSFORMS
from scipy.ndimage import gaussian_filter


@TRANSFORMS.register_module()
class Blur(BaseTransform):
    """Randomly apply Gaussian blur to the image.

    Args:
        sigma (Tuple[float, float]): The range for random sampling the Gaussian blur sigma. 
            The blur will be applied with a random sigma sampled from this range. Defaults to (0.1, 2.0).
    """

    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0)) -> None:
        assert isinstance(sigma, tuple) and len(sigma) == 2, \
            f'sigma should be a tuple with two values, got {sigma}.'
        self.sigma = sigma

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to the image with the given sigma value."""
        sigma_value = random.uniform(self.sigma[0], self.sigma[1])
        # print(f"Applying Gaussian blur with sigma: {sigma_value:.4f}")
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = gaussian_filter(img, sigma=sigma_value)
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = gaussian_filter(img, sigma=sigma_value)
        return img

    def transform(self, results: dict) -> dict:
        """Apply random Gaussian blur to the image."""
        img = results['img']
        results['img'] = self._apply_blur(img)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(sigma={self.sigma}, '
        repr_str += f'seed={self.seed})'
        return repr_str


@TRANSFORMS.register_module()
class SCGTransform(BaseTransform):
    """
    Spectral Causal Generation (SCG)

    Args:
        v1_scale (float): (default 0.005)
        v2_scale (float): (default 0.7)
    """

    def __init__(self, v1_scale: float = 0.005, v2_scale: float = 0.7):
        super().__init__()
        self.v1_scale = v1_scale
        self.v2_scale = v2_scale

    def _apply_scg(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (np.ndarray):  (H, W, C)
        
        Returns:
            np.ndarray:  (H, W, C)
        """
        try:
            h, w = img.shape[:2]

            if h % 2 != 0 or w % 2 != 0:
                img = cv2.resize(img, (w - w % 2, h - h % 2), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]

            img_dct = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                img_dct[:, :, i] = cv2.dct(img[:, :, i].astype(np.float32))

            min_side = min(h, w)
            v1 = max(1, int(min_side * self.v1_scale))
            v2 = max(v1 + 1, int(min_side * self.v2_scale))
            v3 = max(v2 + 1, min_side)

            x_idx, y_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            max_xy = np.maximum(x_idx, y_idx)

            mask = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                cond1 = max_xy <= v1
                mask[:, :, c][cond1] = 1 - max_xy[cond1] / (v1 + 1e-7) * 0.95
                cond2 = (max_xy > v1) & (max_xy <= v2)
                mask[:, :, c][cond2] = 0.01
                cond3 = (max_xy > v2) & (max_xy <= v3)
                mask[:, :, c][cond3] = (max_xy[cond3] - v2) / (v3 - v2 + 1e-7) * 0.3
                cond4 = max_xy > v3
                mask[:, :, c][cond4] = 0.5

            n_mask = 1 - mask
            non_dct = img_dct * mask
            cal_dct = img_dct * n_mask

            ref_dct = np.zeros_like(non_dct)
            for c in range(3):
                noise = random.gauss(0, 1)

                ref_dct[:, :, c] = non_dct[:, :, c] * (1 + noise)

            img_fc = ref_dct + cal_dct
            img_out = np.zeros_like(img_fc)
            for i in range(3):
                img_out[:, :, i] = cv2.idct(img_fc[:, :, i]).clip(0, 255)

            img_out = np.clip(img_out, 0, 255)

            return img_out.astype(np.uint8)

        except Exception as e:
            print(f"SCGTransform failed: {e}")
            return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self._apply_scg(img)
        return results


@TRANSFORMS.register_module()
class MAGTransform(BaseTransform):

    def __init__(self, v1_scale: float = 0.005, v2_scale: float = 0.7):
        super().__init__()
        self.v1_scale = v1_scale
        self.v2_scale = v2_scale

    def _apply_scg(self, img: np.ndarray) -> np.ndarray:
        try:
            h, w = img.shape[:2]

            fft_img = np.zeros((h, w, 3), dtype=np.complex64)
            amp = np.zeros((h, w, 3), dtype=np.float32)
            phase = np.zeros((h, w, 3), dtype=np.float32)

            for c in range(3):
                f = np.fft.fft2(img[:, :, c])
                fshift = np.fft.fftshift(f)
                fft_img[:, :, c] = fshift
                amp[:, :, c] = np.abs(fshift)
                phase[:, :, c] = np.angle(fshift)

            v1 = int(min(h, w) * self.v1_scale)
            v2 = int(min(h, w) * self.v2_scale)
            v3 = min(h, w)

            x_idx, y_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            max_xy = np.maximum(x_idx, y_idx)

            mask = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                cond1 = max_xy <= v1
                mask[:, :, c][cond1] = 1 - max_xy[cond1] / v1 * 0.95
                cond2 = (max_xy > v1) & (max_xy <= v2)
                mask[:, :, c][cond2] = 0.01
                cond3 = (max_xy > v2) & (max_xy <= v3)
                mask[:, :, c][cond3] = (max_xy[cond3] - v2) / (v3 - v2) * 0.3
                cond4 = max_xy > v3
                mask[:, :, c][cond4] = 0.5

            n_mask = 1 - mask
            non_amp = amp * mask
            cal_amp = amp * n_mask

            ref_amp = np.zeros_like(non_amp)
            for c in range(3):
                noise = random.gauss(0, 1)
                ref_amp[:, :, c] = non_amp[:, :, c] * (1 + noise)

            final_amp = ref_amp + cal_amp

            out_img = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                new_fft = final_amp[:, :, c] * np.exp(1j * phase[:, :, c])

                new_fft = np.fft.ifftshift(new_fft)

                img_rec = np.fft.ifft2(new_fft).real
                out_img[:, :, c] = np.clip(img_rec, 0, 255)

            return out_img.astype(np.uint8)

        except Exception as e:
            print(f"MAGTransform FFT failed: {e}")
            return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        img_out = self._apply_scg(img)
        results['img'] = img_out
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(v1_scale={self.v1_scale}, '
        repr_str += f'v2_scale={self.v2_scale})'
        return repr_str


@TRANSFORMS.register_module()
class FFTMagnitudeTransform(BaseTransform):

    def __init__(self, A=1.2, B=0.5, C=0.5):
        self.A = A
        self.B = B if B != 0 else 1e-6
        self.C = C

    def _calc(self, d_n: np.ndarray) -> np.ndarray:
        return (self.A * d_n) / (self.B + d_n) + self.C

    def _apply_mag(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape
        img_out = np.zeros_like(img, dtype=np.uint8)

        cy, cx = h // 2, w // 2

        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - cx)**2 + (Y - cy)**2)

        r_max = max(h / 2, w / 2)

        d_n = d / r_max

        w_2d = self._calc(d_n)

        for i in range(c):
            f = np.fft.fft2(img[:, :, i].astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            phase = np.angle(fshift)

            mag_new = mag * w_2d

            fshift_new = mag_new * np.exp(1j * phase)
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift_new))
            img_back = np.real(img_back)

            img_back -= img_back.min()
            img_back /= img_back.max()
            img_out[:, :, i] = (img_back * 255).astype(np.uint8)

        return img_out

    def transform(self, results: dict) -> dict:
        img = results["img"]
        results["img"] = self._apply_mag(img)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(A={self.A}, B={self.B}, C={self.C})"


@TRANSFORMS.register_module()
class Saturation(BaseTransform):
    """Randomly adjusts the saturation of an RGB image.
    
    Args:
        min_mag (float): Minimum saturation adjustment factor. Defaults to 0.1.
        max_mag (float): Maximum saturation adjustment factor. Defaults to 1.9.
    """

    def __init__(self, min_mag: float = 0.1, max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for RandomSaturation should be in range [0, 2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for RandomSaturation should be in range [0, 2], got {max_mag}.'
        self.min_mag = min_mag
        self.max_mag = max_mag

    def _apply_blend(self, img: np.ndarray, grayscale: np.ndarray, weight: float) -> np.ndarray:
        """Blend the original image with its grayscale version based on the weight."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = grayscale * (1 - weight) + img * weight
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        else:
            return grayscale * (1 - weight) + img * weight

    def _apply_saturation(self, img: np.ndarray) -> np.ndarray:
        """Apply random saturation adjustment to image."""
        assert img.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = random.uniform(self.min_mag, self.max_mag)
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        img_out = self._apply_blend(img, grayscale, w)
        return img_out

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = self._apply_saturation(img)
        return results


@TRANSFORMS.register_module()
class RandomEraseTransform(BaseTransform):
    """
    Random Erasing operation adapted from Detectron2/Torchvision style.

    Random Erasing randomly selects a rectangle region in an image
    and erases its pixels with random values.

    Args:
        sl (float): Minimum proportion of erased area against input image. Default: 0.02.
        sh (float): Maximum proportion of erased area against input image. Default: 0.4.
        r1 (float): Minimum aspect ratio of the erased area. Default: 0.3.
        r2 (float): Maximum aspect ratio of the erased area. Default: 3.3.
        value (str or float): Value to use for erasure. If 'random', it assigns random values.
                              If float or tuple/list, it will use that value for all channels.
    """

    def __init__(self,
                 sl: float = 0.02,
                 sh: float = 0.4,
                 r1: float = 0.3,
                 r2: float = 3.3,
                 value: Union[str, float] = "random") -> None:
        super().__init__()
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.value = value

    def _apply_erase(self, img: np.ndarray) -> np.ndarray:
        was_int = False
        if img.dtype == np.uint8:
            was_int = True
            img = img.astype(np.float32)

        imgh, imgw, c = img.shape
        area = imgw * imgh

        for attempt in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < imgw and h < imgh:
                h0 = random.randint(0, imgh - h)
                w0 = random.randint(0, imgw - w)

                if self.value == "random":
                    patch = torch.rand(h, w, c).numpy()

                    if was_int:
                        patch *= 255

                    img[h0:h0 + h, w0:w0 + w, :] = patch
                else:
                    img[h0:h0 + h, w0:w0 + w, :] = self.value

                break

        if was_int:
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return img

    def transform(self, results: dict) -> dict:
        """Apply random erasing to the image stored in results['img']."""
        img = results['img']
        results['img'] = self._apply_erase(img)
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(sl={self.sl}, sh={self.sh}, "
                f"r1={self.r1}, r2={self.r2}, value={self.value})")


# https://arxiv.org/pdf/2212.01322
@TRANSFORMS.register_module()
class MICTransform(BaseTransform):
    """
    MIC (Masked Image Completion) Transform for MMDetection.

    This transform randomly masks out patches in the image based on a given ratio
    and block size. The masked image is then used for training.

    Args:
        ratio (float): The probability of a block being erased. Should be in range [0, 1].
        block_size (int): Size of the blocks to be used for masking (height and width).
    """

    def __init__(self, ratio: float, block_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.block_size = block_size

    def _apply_mic(self, img: np.ndarray) -> np.ndarray:
        """Apply MIC transformation to the image."""
        H, W, C = img.shape
        mh, mw = round(H / self.block_size), round(W / self.block_size)

        input_mask = torch.rand(mh, mw).numpy()

        input_mask = input_mask > self.ratio

        input_mask = cv2.resize(input_mask.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST)

        masked_img = img * input_mask[..., np.newaxis]

        return masked_img.astype(img.dtype)

    def transform(self, results: dict) -> dict:
        """Main transform function to apply MIC transformation."""
        img = results['img']
        results['img'] = self._apply_mic(img)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(ratio={self.ratio}, block_size={self.block_size})'


@TRANSFORMS.register_module()
class KeepGTOnly(BaseTransform):
    """Keep only the GT regions; mask out the rest with black.

    This transform replaces all pixels outside GT bboxes with zeros.
    """

    def __init__(self) -> None:
        pass

    def _mask_background(self, img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Zero out everything outside GT boxes."""
        h, w = img.shape[:2]

        masked = np.zeros_like(img)
        boxes = bboxes.tensor.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            x2 = np.clip(x2, 1, w)
            y2 = np.clip(y2, 1, h)

            masked[y1:y2, x1:x2] = img[y1:y2, x1:x2]

        return masked

    def transform(self, results: dict) -> dict:
        """Apply mask so that only GT areas remain."""
        img = results['img']

        bboxes = results.get('gt_bboxes', None)
        if bboxes is None or len(bboxes) == 0:
            results['img'] = np.zeros_like(img)
            return results

        results['img'] = self._mask_background(img, bboxes)

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class StructureAugment(BaseTransform):
    """
    Enhance the structural features of the image by injecting edge information.

    Args:
        prob (float): Probability of applying the transformation.
        alpha_range (tuple): Range of edge injection strength (min, max).
                         e.g., (0.5, 1.5). Larger values produce more pronounced edges.
        mode (str): 'sobel' or 'shift'.
    """

    def __init__(self, prob=0.5, alpha_range=(0.3, 1.0), mode='sobel'):
        super().__init__()
        self.prob = prob
        self.alpha_range = alpha_range
        self.mode = mode

    def _get_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.mode == 'sobel':
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            abs_x = cv2.convertScaleAbs(grad_x)
            abs_y = cv2.convertScaleAbs(grad_y)
            edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

        elif self.mode == 'shift':
            padded = np.pad(gray, ((0, 1), (0, 1)), mode='edge')
            shifted = padded[1:, 1:]
            edge = cv2.absdiff(gray, shifted)

        else:
            raise NotImplementedError

        edge_3c = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        return edge_3c

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        img = results['img']  # uint8

        edges = self._get_edges(img)

        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

        img_float = img.astype(np.float32)
        edge_float = edges.astype(np.float32)

        img_aug = img_float + alpha * edge_float

        img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        results['img'] = img_aug
        return results


@TRANSFORMS.register_module()
class RetinexNorm(BaseTransform):
    """
    Illumination normalization based on Retinex theory.
    Enhances robustness under varying lighting conditions by removing low-frequency illumination components.
    Computationally expensive.
    """

    def __init__(self, sigma_list=[15, 80, 250], prob=0.5):
        super().__init__()
        self.sigma_list = sigma_list  # Multi-scale Gaussian kernels
        self.prob = prob

    def _single_scale_retinex(self, img, sigma):
        # L = Gaussian(I)
        retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
        return retinex

    def _single_scale_retinex_fast(self, img, sigma):
        # Accelerated Single Scale Retinex
        # 1. Downsample - reduce to e.g. 1/4 size or a fixed smaller resolution
        # Compute scaling factor, e.g. resize so the shorter side becomes 256 pixels
        h, w = img.shape[:2]
        scale = 256.0 / min(h, w)
        if scale < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # Sigma should also be scaled accordingly, otherwise blur strength becomes incorrect
            small_sigma = sigma * scale
        else:
            small_img = img
            small_sigma = sigma

        # 2. Apply Gaussian blur on the smaller image (very fast)
        blur_small = cv2.GaussianBlur(small_img, (0, 0), small_sigma)

        # 3. Upsample back to original resolution to obtain illumination estimation L
        blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Perform log subtraction at original resolution
        # Note: log on the original image is still required and cannot be avoided
        return np.log10(img + 1.0) - np.log10(blur + 1.0)

    def _multi_scale_retinex(self, img):
        retinex = np.zeros_like(img, dtype=np.float32)
        for sigma in self.sigma_list:
            retinex += self._single_scale_retinex_fast(img, sigma)
        retinex = retinex / len(self.sigma_list)
        return retinex

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        img = results['img'].astype(np.float32)

        # 1. Compute Log(R)
        # The result is typically in a small range around [-1, 1]
        m_retinex = self._multi_scale_retinex(img)

        # 2. Normalize back to [0, 255] for further processing
        # Requires statistical normalization to map values from log domain to image domain
        # Simple Min-Max normalization may be sensitive to noise
        # Here we use mean-std normalization
        for i in range(3):
            mean = np.mean(m_retinex[:, :, i])
            std = np.std(m_retinex[:, :, i])
            # Clamp within ±2 std around mean and stretch contrast
            m_retinex[:, :, i] = (m_retinex[:, :, i] - mean) / (std + 1e-6) * 50 + 128

        img_out = np.clip(m_retinex, 0, 255).astype(np.uint8)

        results['img'] = img_out
        return results
