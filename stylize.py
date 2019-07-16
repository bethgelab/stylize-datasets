#!/usr/bin/env python
import argparse
from pathlib import Path
from PIL import Image
import random
from torchvision.utils import save_image
from tqdm import tqdm
from Transferer import StyleTransfer

parser = argparse.ArgumentParser(description='This script applies the AdaIN style transfer method to arbitrary image-containing directories.')
parser.add_argument('--content-dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style-dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save the output images')
parser.add_argument('--num-styles', type=int, default=1, help='Number of styles to \
                        create for each image (default: 1)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                          stylization. Should be between 0 and 1')
parser.add_argument('--extensions', nargs='+', type=str, default=['png', 'jpeg', 'jpg'], help='List of image extensions to scan style and content directory for (case sensitive), default: png, jpeg, jpg')

# Advanced options
parser.add_argument('--content-size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')

# random.seed(131213)
def main():
    args = parser.parse_args()

    # set content and style directories
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    style_dir = style_dir.resolve()
    output_dir = Path(args.output_dir)
    output_dir = output_dir.resolve()
    assert style_dir.is_dir(), 'Style directory not found'

    # collect content files
    extensions = args.extensions
    assert len(extensions) > 0, 'No file extensions specified'
    content_dir = Path(content_dir)
    content_dir = content_dir.resolve()
    assert content_dir.is_dir(), 'Content directory not found'
    dataset = []
    for ext in extensions:
        dataset += list(content_dir.rglob('*.' + ext))

    assert len(dataset) > 0, 'No images with specified extensions found in content directory' + content_dir
    content_paths = sorted(dataset)
    print('Found %d content images in %s' % (len(content_paths), content_dir))

    # collect style files
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    ST = StyleTransfer(alpha=args.alpha, content_size=args.content_size, style_size=args.style_size, crop=args.crop)

    # actual style transfer as in AdaIN
    with tqdm(total=len(content_paths) * args.num_styles) as pbar:
        for content_path in content_paths:
            try:
                content_img = Image.open(content_path).convert('RGB')
            except OSError as e:
                print('Skipping stylization of %s due to error below' %(content_path))
                print(e)
                continue
            for style_path in random.sample(styles, args.num_styles):
                try:
                    style_img = Image.open(style_path).convert('RGB')
                except OSError as e:
                    print('Skipping stylization of %s with %s due to error below' %(content_path, style_path))
                    print(e)
                    continue

                stylized = ST.stylize(content_img, style_img)

                rel_path = content_path.relative_to(content_dir)
                out_dir = output_dir.joinpath(rel_path.parent)

                # create directory structure if it does not exist
                if not out_dir.is_dir():
                    out_dir.mkdir(parents=True)

                content_name = content_path.stem
                style_name = style_path.stem
                out_filename = content_name + '-stylized-' + style_name + content_path.suffix
                output_name = out_dir.joinpath(out_filename)

                save_image(stylized, output_name, padding=0) #default image padding is 2.
                style_img.close()
                pbar.update(1)
            content_img.close()

if __name__ == '__main__':
    main()
