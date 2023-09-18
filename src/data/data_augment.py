import logging
import random
from PIL import Image
# from albumentations import (
#     HorizontalFlip, VerticalFlip, IAAPerspective, CLAHE, RandomRotate90, Rotate,
#     Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
#     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, CoarseDropout,
#     ShiftScaleRotate, CenterCrop, Resize, ColorJitter, RandomBrightness, ToGray,
#     RandomScale, GridDropout, ElasticTransform, RandomGridShuffle, ChannelShuffle, ChannelDropout, JpegCompression
# )
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import math
import numpy as np
import torch
from torchvision.transforms import ColorJitter
from src.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from src.data.transforms import _pil_interp
from src.data.random_erasing_arr import RandomErasing
from copy import deepcopy


def get_train_transforms(args, img_size, resize_type="letterbox", interpolation=cv2.INTER_CUBIC):
    if isinstance(img_size, (list, tuple)):
        height, width = img_size
    elif isinstance(img_size, int):
        height, width = img_size, img_size
    else:
        print("error img size:{}".format(img_size))

    if resize_type == "letterbox":
        train_transforms = [
            LetterBoxResize(height=height, width=width, interpolation=interpolation, flip_prob=1.0, flips=args.train_flips)]
    elif resize_type == "normal":
        train_transforms = [
            A.RandomResizedCrop(height, width,
                                scale=(0.3, 1.0),
                                ratio=(0.75, 1.34),
                                interpolation=interpolation)
            # A.Resize(height, width, interpolation=interpolation)
            ]
    else:
        print("error resize type: {}".format(resize_type))
    
    if args.aa is not None:
        train_transforms.append(MyAutoAugment(img_size, interpolation, args.aa))
    
    if args.border_mode == "constant":
        border_mode = cv2.BORDER_CONSTANT
    elif args.border_mode == "reflect":
        border_mode = cv2.BORDER_REFLECT_101
    else:
        print("error args.border_mode:{}".format(args.border_mode))
    hole_wh = height * 16 // 224
    # hole_wh = 16
    train_transforms.extend([
        # Transpose(p=0.5),
        A.HorizontalFlip(p=args.hflip), # 沿Y轴
        # A.VerticalFlip(p=0.5), # 沿X轴 0625
        A.ShiftScaleRotate(shift_limit=args.shift_limit, scale_limit=args.scale_limit, rotate_limit=args.rotate_limit, 
                         border_mode=border_mode, interpolation=interpolation, p=args.ssrprob),
        A.GridDropout(ratio=args.gdratio, p=args.gdprob), # 0.3-0.4之间
        # A.ToGray(p=0.5),
        # alpha越小，sigma越大，产生的偏差越小，和原图越接近
        # ElasticTransform(alpha=1, sigma=100, alpha_affine=10, 
                        #  border_mode=cv2.BORDER_REFLECT_101, interpolation=interpolation, p=1.0), 
        A.RandomGridShuffle(grid=(args.rgsgrid, args.rgsgrid), p=args.rgsprob),
        # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0)], p=0.5),
        # A.OneOf([
        #     A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
        #     A.ChannelShuffle(p=0.5)], p=0.5),
        # A.OneOf([
        #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
        #     A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5)], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5), 
            A.MotionBlur(blur_limit=5, p=0.5), 
            A.MedianBlur(blur_limit=5, p=0.5)], p=0.5),
        # A.CoarseDropout(max_height=hole_wh, max_width=hole_wh, max_holes=8, p=0.5),
        AlbuYoloCutout(p=args.ycprob),
        # 尝试新数据增强
        # A.ChannelShuffle(p=0.5),
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0), 
        # RandomPasteCurImg(dst_size=64),
        ToTensorV2(p=1.0),
        AlbuRandomErasingv3(mode=args.remode, min_area=args.min_area, max_area=args.max_area, 
                            max_count=args.recount, device='cpu', p=args.reprob)
    ])
    return A.Compose(train_transforms, p=1.)


def get_tta_transforms(img_size, interpolation=cv2.INTER_CUBIC):
    if isinstance(img_size, (list, tuple)):
        if len(img_size) == 1:
            height, width = img_size[0], img_size[0]
        else:
            height, width = img_size
    elif isinstance(img_size, int):
        height, width = img_size, img_size
    else:
        print("error img size:{}".format(img_size))
        
    valid_transforms = [
        A.Resize(height, width, interpolation=interpolation),
    ]

    valid_transforms.extend([
        A.HorizontalFlip(p=0.5),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, 
                        #  border_mode=cv2.BORDER_CONSTANT, interpolation=interpolation, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    return A.Compose(valid_transforms, p=1.)

def get_valid_transforms(img_size, resize_type="letterbox", crop_pct=0.95, interpolation=cv2.INTER_CUBIC, flips=[0,1]):
    if isinstance(img_size, (list, tuple)):
        if len(img_size) == 1:
            height, width = img_size[0], img_size[0]
        else:
            height, width = img_size
    elif isinstance(img_size, int):
        height, width = img_size, img_size
    else:
        print("error img size:{}".format(img_size))
        
    if resize_type == "letterbox":
        valid_transforms = [
            LetterBoxResize(height=height, width=width, interpolation=interpolation, flips=flips)
        ]
    elif resize_type == "normal":
        # scale_size = int(math.floor(height / crop_pct))
        # valid_transforms = [
        #     A.Resize(scale_size, scale_size, interpolation=interpolation),
        #     A.CenterCrop(img_size, img_size),
        # ]
        valid_transforms = [
            A.Resize(height, width, interpolation=interpolation),
        ]
    else:
        print("error resize type: {}".format(resize_type))

    valid_transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    return A.Compose(valid_transforms, p=1.)

class Cv2Resize(DualTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(Cv2Resize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, cv2_img, **params):
        resize_img = cv2.resize(cv2_img, (self.width, self.height), self.interpolation)
        return resize_img

class LetterBoxResize(DualTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_CUBIC, always_apply=False, flip_prob=1.0, flips=[0, 1], p=1):
        super(LetterBoxResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.flip_prob = flip_prob
        self.flips = flips

    def apply(self, cv2_img, **params):
        new_width, new_height = self.width, self.height

        if len(self.flips) >= 1:
            flip_num = np.random.choice(self.flips, 1)[0]
            
            if flip_num != 2:
                flip_img = cv2.flip(deepcopy(cv2_img), flip_num)
                cv2_img = np.hstack((cv2_img, flip_img))
            else:
                cv2_img = np.hstack((cv2_img, cv2_img))
        else:
            cv2_img = cv2_img
        
        ori_height, ori_width = cv2_img.shape[:2]

        resize_ratio = min(new_width / ori_width, new_height / ori_height)

        resize_w = int(resize_ratio * ori_width)
        resize_h = int(resize_ratio * ori_height)

        img = cv2.resize(cv2_img, (resize_w, resize_h), self.interpolation)
        image_padded = np.full((new_height, new_width, 3), 0, np.uint8)

        dw = int((new_width - resize_w) / 2)
        dh = int((new_height - resize_h) / 2)

        image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

        return image_padded

class MyAutoAugment(DualTransform):
    def __init__(self, img_size, interpolation, policy, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.img_size = img_size
        self.interpolation = interpolation
        self.policy = policy
        
    def apply(self, img, **params):
        mean = (0.485, 0.456, 0.406)
        if isinstance(self.img_size, tuple):
            img_size_min = min(self.img_size)
        else:
            img_size_min = self.img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if self.interpolation and self.interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(self.interpolation)
        if self.policy.startswith('rand'):
            augment_transform = rand_augment_transform(self.policy, aa_params)
        elif self.policy.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            augment_transform = augment_and_mix_transform(self.policy, aa_params)
        else:
            augment_transform = auto_augment_transform(self.policy, aa_params)
        
        pil_img = Image.fromarray(img)
        aug_img = augment_transform(pil_img)
        return np.asarray(aug_img)

class AlbuRandomErasing(DualTransform):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465), always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def apply(self, img, **params):

        if random.uniform(0, 1) >= self.p:
            return img

        ih, iw, ic = img.shape
        for _ in range(100):
            area = ih * iw

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < iw and h < ih:
                x1 = random.randint(0, ih - h)
                y1 = random.randint(0, iw - w)
                if ic == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

class AlbuGridMask(DualTransform):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, always_apply=False, prob=1.):
        super().__init__(always_apply, prob)
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def apply(self, img, **params):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        d = np.random.randint(self.d1, self.d2)
        #d = self.d
        
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask

        mask = mask.expand_as(img)
        img = img * mask 

        return img

class AlbuRandomErasingv3(DualTransform):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda', always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.p = p
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    @staticmethod
    def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
        # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
        # paths, flip the order so normal is run on CPU if this becomes a problem
        # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
        if per_pixel:
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif rand_color:
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
    
    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.p:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = self._get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def apply(self, input, **params):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

class AlbuRandomErasingv2(DualTransform):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.p = p
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        
    def _erase(self, img):
        img_h, img_w, chan = img.shape
        dtype = img.dtype
        if random.random() > self.p:
            return img
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[top:top + h, left:left + w, :] = self._get_pixels(
                        self.per_pixel, self.rand_color, (h, w, chan),
                        dtype=dtype)
                    break
        return img
    
    @staticmethod
    def _get_pixels(per_pixel, rand_color, patch_size, dtype=np.float32):
        mean = np.array([[[0, 0, 0]]])
        std = np.array([[1, 1, 1]])
        if per_pixel:
            return (np.empty(patch_size, dtype=dtype) - mean) / std
        elif rand_color:
            return (np.empty((patch_size[0], 1, 1), dtype=dtype) - mean) / std
        else:
            return np.zeros((patch_size[0], 1, 1), dtype=dtype)
    
    def apply(self, cv2_img, **params):
        aug_img = self._erase(cv2_img)
        return aug_img

class AlbuYoloCutout(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.p = p
        
    def cutout(self, img):
        if random.random() > self.p:
            return img
        h, w = img.shape[:2]

        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        return img

    def apply(self, cv2_img, **params):
        aug_img = self.cutout(cv2_img)
        return aug_img
    
class RandomPasteCurImg(DualTransform):
    def __init__(self, cur_img, dst_size=32, bbox=None, outer_box=False, always_apply=False, p=1):
        super(RandomPasteCurImg, self).__init__(always_apply, p)
        self.cur_img = cur_img
        self.dst_size = dst_size
        self.bbox = bbox
        self.outer_box = outer_box

    def find_contours(self, img):
        if img.shape[-1] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        cnts, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        index = np.argmax([cv2.contourArea(c) for c in cnts])

        return cnts[index]

    def isPointWithinBox(self, point, sbox):
        # sbox=[x, y, w, h]
        px, py = point[0], point[1]
        x1, y1, x2, y2 = sbox[0], sbox[1], sbox[0] + sbox[2], sbox[1] + sbox[3]
        # 不考虑在边界上，需要考虑就加等号
        if x1 < px < x2 and y1 < py < y2:
            return True
        return False

    def apply(self, img, **params):
        cur_img = self.cur_img
        dst_size = self.dst_size
        raw_img = self.raw_img
        outer_box = self.outer_box
        bbox = self.bbox
        if cur_img.shape[-1] != 3:
            # 对原始部分光标文件做处理
            cur_img = np.where(cur_img == 1, 255, cur_img)
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2BGR)
        # 将光标图像缩放到指定大小， [32, 64, 96, 128]
        cur_img = cv2.resize(cur_img, (dst_size, dst_size), cv2.INTER_CUBIC)

        raw_img_arr = np.asarray(raw_img, dtype=np.uint8)
        cur_img_arr = np.asarray(cur_img, dtype=np.uint8)

        rh, rw = raw_img_arr.shape[:-1]
        ch, cw = cur_img_arr.shape[:-1]
        ph, pw = ch // 2, cw // 2

        if not outer_box and ch <= bbox[3] and cw <= bbox[2]:
            x_min, x_max, y_min, y_max = bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]
            # 将 cx, cy 限制在 [pw, rw - pw] 和 [ph, rh - ph] 内
            if x_min < pw: x_min = pw
            if y_min < ph: y_min = ph
            if x_max > rw - pw: x_max = rw - pw
            if y_max > rh - ph: y_max = rh - ph
        else:
            x_min, x_max, y_min, y_max = pw, rw - pw, ph, rh - ph

        cx, cy = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

        if outer_box:
            flag = True
            while flag:
                cx, cy = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)
                if not self.isPointWithinBox((cx, cy), bbox):
                    flag = False

        # 对待粘贴的光标区域图像中的黑色区域赋值为原始图像，其他区域为光标
        crop_img = raw_img_arr[cy - ph:cy + ph, cx - pw:cx + pw]
        crop_img = np.where(cur_img_arr == 0, crop_img, cur_img_arr)

        cnt = self.find_contours(cur_img)
        cv2.drawContours(crop_img, cnt, -1, (0, 0, 0), 1)

        # 把处理好的局部图粘贴到原始图像中
        raw_img_arr[cy - ph:cy + ph, cx - pw:cx + pw] = crop_img
        return raw_img_arr
