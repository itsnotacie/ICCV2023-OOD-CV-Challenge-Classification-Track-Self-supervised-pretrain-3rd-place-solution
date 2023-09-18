import torch.utils.data
import numpy as np

from .transforms_factory import create_transform
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .distributed_sampler import OrderedDistributedSampler
from .random_erasing import RandomErasing
from .mixup_ori import FastCollateMixup


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


class batch_sampler():
    def __init__(self, batch_size, class_list):
        self.batch_size = batch_size
        self.class_list = class_list
        self.unique_value = np.unique(class_list)
        self.iter_list = []
        self.few_list = []
        temp_list = []
        # before every batch shuffle class list
        np.random.shuffle(self.class_list)
        for idx, v in enumerate(self.unique_value):
            # find the class index in the total class_list
            indexes = [i for i, x in enumerate(self.class_list) if x == v]
            temp_list.append([len(indexes), indexes])
            # if idx == 0:
            #     # 将few类的index单独保存起来
            #     self.few_list = indexes
            # else:
            #     # 保存每个类别的数量和indexes
            #     temp_list.append([len(indexes), indexes])
        # 对每个类别的数量进行从大到小排序，使得batch-size超过整数倍的类别数，能多取一些多类的图片
        temp_list = sorted(temp_list, key=lambda k: k[0], reverse=True)
        # max_class_number = temp_list[0][0]
        for lst in temp_list:
            self.iter_list.append(self.shuffle_iterator(lst[-1]))
        self.batch_num = len(self.class_list) // self.batch_size
        # 通过计算每个batch中每个类的图像数量，计算迭代完需要的batch数量, 经过测试发现对coating有提升，对morph分类效果不好
        # self.batch_num = max_class_number // (self.batch_size // len(self.unique_value))

    def __iter__(self):
        index_list = []
        for i in range(self.batch_num):
            # 对每个batch的数据进行重新选择
            for index in range(self.batch_size):
                idx_list = next(self.iter_list[index % (len(self.unique_value))])
                for idx in idx_list:
                    index_list.append(idx)
            # 单独将少类的图像index加入batch中
            # index_list.extend(self.few_list)
            np.random.shuffle(index_list)
            yield index_list
            index_list = []

    def __len__(self):
        return self.batch_num

    @staticmethod
    def shuffle_iterator(iterator, step=1):
        # iterator should have limited size
        index = list(iterator)
        total_size = len(index)
        i = 0
        np.random.shuffle(index)
        while True:
            yield index[i:i + step]
            i += step
            if i + step >= total_size:
                i = 0
                np.random.shuffle(index)


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0):
        self.loader = loader
        self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    next_input = next_input.half().sub_(self.mean).div_(self.std)
                else:
                    next_input = next_input.float().sub_(self.mean).div_(self.std)
                if self.random_erasing is not None:
                    next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x


def create_loader(
        dataset,
        args,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    if is_training and args.use_imbalance_sampler:
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler(batch_size, dataset.labels),
            collate_fn=collate_fn,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None and is_training,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=is_training,
        )

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=re_prob if is_training else 0.,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader



