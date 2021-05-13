import torch.utils.data as data

from PIL import Image #图像处理

import os
import os.path


def has_file_allowed_extension(filename, extensions): #检查扩展名
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions) 
    # extensions里有的才接受，any是如果有1个true则返回true，只有全是false才返回false。只要文件名结尾再extension里面，就是对的


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] #如果拼接后是个文件夹名称则加入classes #listdir遍历文件夹一层
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx # 对每张图片返回类名和索引编号


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)): #walk把文件夹内容遍历出来，root是外面那层目录
        for fname in sorted(fnames): #fname是图片的文件名
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname) #如果文件名对，则把文件夹和文件拼接
                item = (path, 0)
                images.append(item)

    return images


class DatasetFolder(data.Dataset): #继承pytorch的dataset类
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, extensions) #把root里所有图片读出来
        if len(samples) == 0: #如果没读出来，则报错
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] #(path, 0)
        sample = self.loader(path) #打开图片转RGB
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples) #张量

    def __repr__(self): #描述
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: #rb二进制
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform) #继承父类变量
        self.imgs = self.samples
