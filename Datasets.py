from pathlib import Path
import glob
import shutil
from Transferer import StyleTransfer
from tqdm import tqdm
import random
from PIL import Image
from torchvision.utils import save_image
import sys
from importlib import import_module


class StylizeDataset():
    def __init__(self,
                dataset_dir=None,
                target_dir=None,
                style_dir=None,
                whitelist=None,
                blacklist=None,
                image_dirs=None,
                content_extensions=None,
                style_extensions=None,
                alpha=1.0,
                content_size=0,
                style_size=512,
                crop=False,
                seed=None):
        """
        TODO
        """
        assert type(dataset_dir) == str, 'dataset_dir must be a string'
        assert type(target_dir) == str, 'target_dir must be a string'
        assert type(style_dir) == str, 'target_dir must be a string'
        dataset_path = Path(dataset_dir).resolve()
        assert dataset_path.exists(), '%s does not exist.' % dataset_dir
        self.dataset_dir = dataset_path

        style_path = Path(style_dir).resolve()
        assert style_path.exists(), '%s does not exist.' % style_dir
        self.style_dir = style_path

        target_path = Path(target_dir).resolve()
        self.target_dir = target_path

        self.whitelist = None
        self.blacklist = None

        if not (whitelist is None):
            if type(whitelist) == str:
                whitelist = [whitelist]
            self.whitelist = whitelist
        elif not (blacklist is None):
            if type(blacklist) == str:
                blacklist = [blacklist]
            self.blacklist = blacklist

        if not(image_dirs is None):
            if type(image_dirs) == str:
                image_dirs = [image_dirs]
            self.image_dirs = image_dirs
            for imgdir in self.image_dirs:
                assert self.dataset_dir.joinpath(imgdir).is_dir() and (self.blacklist is None or not (imgdir in self.blacklist))
        else:
            self.image_dirs = None

        if type(content_extensions) == str:
            content_extensions = [content_extensions]
        elif content_extensions is None:
            content_extensions = ['jpg', 'jpeg', 'png']
        self.content_extensions = content_extensions
        if type(style_extensions) == str:
            style_extensions = [style_extensions]
        elif style_extensions is None:
            style_extensions = ['jpg', 'jpeg', 'png']
        self.style_extensions = style_extensions

        self.ST = StyleTransfer(alpha, content_size, style_size, crop)

        self.seed = seed

        self.copy()
        self.stylize()

    def copy(self):
        if not (self.target_dir.exists()):
            self.target_dir.mkdir()
        if not (self.whitelist is None):
            # copy all subfolders of dataset_dir listed in whitelist
            for entry in self.whitelist:
                dir_or_file = self.dataset_dir.joinpath(entry)
                if dir_or_file.is_file():
                    shutil.copy(dir_or_file, self.target_dir.joinpath(entry))
                elif dir_or_file.is_dir():
                    shutil.copytree(dir_or_file, self.target_dir.joinpath(entry))
                else:
                    print('Skipping whitelisted entry %s because no such file or directory exists in %s' % (entry, self.dataset_dir))
        else:
            # copy all subfolders of dataset_dir except the ones listed in blacklist
            contents = glob.glob(str(self.dataset_dir.joinpath('*')))
            for entry in contents:
                entry = str(Path(entry).relative_to(self.dataset_dir))
                if not(self.blacklist is None) and entry in self.blacklist:
                    continue
                else:
                    dir_or_file = self.dataset_dir.joinpath(entry)
                    if dir_or_file.is_file():
                        shutil.copy(dir_or_file, self.target_dir.joinpath(entry))
                    elif dir_or_file.is_dir():
                        shutil.copytree(dir_or_file, self.target_dir.joinpath(entry))

    def stylize(self):
        if not (self.seed is None):
            random.seed(self.seed)
        # collect contents
        dataset = []
        if not(self.image_dirs is None):
            # only search for images in whitelisted subdirs
            for imgdir in self.image_dirs:
                for ext in self.content_extensions:
                    print(self.target_dir.joinpath(imgdir))
                    dataset += self.target_dir.joinpath(imgdir).rglob('*.' + ext)
        else:
            # search for images in target_dir
            for ext in self.content_extensions:
                dataset += self.target_dir.rglob('*.' + ext)

        assert len(dataset) > 0, 'No images with specified extensions found in dataset directory'
        content_paths = sorted(dataset)
        print('Found %d content images' %(len(content_paths)))

        # collect styles
        styles = []
        for ext in self.style_extensions:
            styles += self.style_dir.rglob('*.' + ext)
        styles = sorted(styles)
        print('Found %d style images' %(len(styles)))

        for content_path in tqdm(content_paths):
            style_img = None
            style_path = random.choice(styles)
            while style_img is None:
                try:
                    style_img = Image.open(style_path).convert('RGB')
                except:
                    pass
            try:
                content_img = Image.open(content_path).convert('RGB')
            except OSError as e:
                print('Skipping stylization of %s due to error below' %content_path)
                print(e)
                continue

            stylized = self.ST.stylize(content_img, style_img)

            save_image(stylized, content_path, padding=0)
            style_img.close()
            content_img.close()

def stylize_dataset_from_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError('File %s does not exist.' %(str(path)))
    module_name = path.stem
    config_dir = path.parent
    sys.path.insert(0, str(config_dir))
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value for name, value in mod.__dict__.items() if not name.startswith('__')
    }
    StylizeDataset(**cfg_dict)
