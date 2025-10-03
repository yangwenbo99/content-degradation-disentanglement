import os
from typing import Dict, Optional, List, Union
import json
import shutil

from torch import Tensor

import dominate
from dominate.tags import a, br, h3, img, meta, p, table, td, tr

from utility import mkdirs, save_image, tensor2im


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400, additional_txts=None):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                if additional_txts is not None:
                    for im, txt1, txt2, link in zip(ims, txts, additional_txts, links):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt1)
                                p(txt2)
                else:
                    for im, txt, link in zip(ims, txts, links):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt)

    def save(self, prefix=''):
        """save the current content to the HMTL file"""
        html_file = '%s/%sindex.html' % (self.web_dir, prefix)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
                                 -- The images shoud have the shape (b, 3, h, w), but only the first image will be saved.
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    name = image_path[0].replace("/", "_")

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, checkpoints_dir, name, web_dir='web'):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.web_dir = os.path.join(checkpoints_dir, web_dir)
        self.img_dir = os.path.join(self.web_dir, 'images')
        self.name = name
        print('[*] create web directory %s...' % self.web_dir)
        mkdirs([self.web_dir, self.img_dir])

    def display_current_results(
            self, visuals: Dict[str, Tensor], 
            epoch: int, global_step: int = -1,
            original_img_path: Optional[Dict[str, str]] = None, 
            bpp: Optional[Dict[str, int]] = None, 
            permanent: bool = False,
            prefix=''):
        """Save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
            permanent (bool) - - if False, the results will be be overwritten by the next call to this function.

        Note: Many sure everything, except visuals, is a scalar or a string.
        """
        saved_image_name = { }
        # save images to the disk
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            suppli_str = f"_step{global_step}" if global_step >= 0 else ""
            this_saved_image_name = f"{prefix}{epoch:03d}{suppli_str}_{label}.png"
            img_path = os.path.join(self.img_dir, this_saved_image_name)
            save_image(image_numpy, img_path)
            saved_image_name[label] = this_saved_image_name

        # It would be better to use a more efficient implementation, but we will be fine. 
        # In fact, this code is already several times more efficient then the
        # original. 
        json_path = os.path.join(self.web_dir, prefix + 'index.json')
        # Check whether the JSON file exists
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                json.dump([], f)
        
        # Load the JSON file
        with open(json_path, 'r') as f:
            image_info: List[Dict[str, Union[str, Dict[str, str]]]] = json.load(f)
            # Image info should have the following structure:
            # image_info[n] = {
            #    'epoch': epoch,
            #    'global_step': global_step,
            #    'saved_image_name': saved_image_name,    # Should have been plural, but the original code uses singular, and I don't want to change it
            #    'original_image_path': original_img_path, (Optional)  # Same
            #    'bpp': bpp, (Optional)   # Same
            # }

        if len(image_info) > 0 and not image_info[-1]['permanent']:
            # If the last image is not permanent, remove it
            last_image_info = image_info.pop()
            for k in last_image_info['saved_image_name'].values():
                os.remove(os.path.join(self.img_dir, k))
        image_info.append({
            'epoch': epoch,
            'global_step': global_step,
            'saved_image_name': saved_image_name,
            'original_image_path': original_img_path,
            'bpp': bpp,
            'permanent': permanent,
            })

        # Save the JSON file
        with open(json_path, 'w') as f:
            json.dump(image_info, f)


    def save_test_results(self, visuals, dis_path, gt_path):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        def get_img_path():
            if label == "gt":
                img_path = os.path.join(self.img_dir, gt_path)
            elif label == "image":
                img_path = os.path.join(self.img_dir, dis_path)
            elif label == "restored_im":
                img_path = os.path.join(self.img_dir, "restored_" + dis_path)
            elif label == "compress_im":
                img_path = os.path.join(self.img_dir, "compress_" + dis_path)
            elif label == "compress_im_map":
                img_path = os.path.join(self.img_dir, "compress_map_" + dis_path)
            elif label == "rec_im":
                img_path = os.path.join(self.img_dir, "rec_" + dis_path)
            elif label == "rec_im_compress":
                img_path = os.path.join(self.img_dir, "rec_compress_" + dis_path)
            elif label == "A_rec_im":
                img_path = os.path.join(self.img_dir, "A_rec_" + dis_path)
            else:
                raise ValueError(f"label {label} is invalid")

            if os.path.splitext(img_path)[-1].lower() in [".tif", ".tiff", ".bmp"]:
                img_path = os.path.splitext(img_path)[0] + ".png"

            return img_path

        # save images to the disk
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            img_path = get_img_path()
            save_image(image_numpy, img_path)

    def display_test_results(self, epoch, dis_paths, gt_paths,
                             iqa_name=None, scores_given=None, scores_rec=None, scores_rec_compress=None, scores_A_rec=None,
                             bpp=None):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        def get_img_path(label, dis_path, gt_path):
            if label == "gt":
                img_path = gt_path
            elif label == "image":
                img_path = dis_path
            elif label == "restored_im":
                img_path = "restored_" + dis_path
            elif label == "compress_im":
                img_path = "compress_" + dis_path
            elif label == "compress_im_map":
                img_path = "compress_map_" + dis_path
            elif label == "rec_im":
                img_path = "rec_" + dis_path
            elif label == "rec_im_compress":
                img_path = "rec_compress_" + dis_path
            elif label == "A_rec_im":
                img_path = "A_rec_" + dis_path
            else:
                raise ValueError(f"label {label} is invalid")

            if os.path.splitext(img_path)[-1].lower() in [".tif", ".tiff", ".bmp"]:
                img_path = os.path.splitext(img_path)[0] + ".png"

            return img_path

        labels = ["gt", "image", "restored_im", "compress_im", "compress_im_map", "rec_im"]
        if scores_rec_compress is not None:
            labels.append("rec_im_compress")
        if scores_A_rec is not None:
            labels.append("A_rec_im")

        # update website
        webpage = HTML(self.web_dir, f"Experiment name = {self.name}, epoch = {epoch}", refresh=0)
        for i in range(len(dis_paths)):
            webpage.add_header(f"image {i + 1}/{len(dis_paths)}: {dis_paths[i]}")
            ims, txts1, links = [], [], []
            if scores_given is not None:
                txts2 = []

            for label in labels:
                img_path = get_img_path(label, dis_paths[i], gt_paths[i])
                ims.append(img_path)

                txt1 = label
                txt1 += f" ({img_path})"
                txts1.append(txt1)

                if label == "gt" or label == "restored_im":
                    txts2.append('')
                if scores_given is not None and label == "image":
                    txts2.append('(' + ', '.join([f"{iqa_name[j]}: {scores_given[i, j]:.3f}" for j in range(len(iqa_name))]) + ')')
                if scores_rec is not None and label == "rec_im":
                    txts2.append('(' + ', '.join([f"{iqa_name[j]}: {scores_rec[i, j]:.3f}" for j in range(len(iqa_name))]) + ')')
                if scores_rec_compress is not None and label == "rec_im_compress":
                    txts2.append('(' + ', '.join([f"{iqa_name[j]}: {scores_rec_compress[i, j]:.3f}" for j in range(len(iqa_name))]) + ')')
                if bpp is not None and label == "compress_im":
                    txts2.append(f"bpp: {bpp[i]:.3f}")

                if scores_A_rec is not None and label == "A_rec_im":
                    txts2.append('(' + ', '.join([f"{iqa_name[j]}: {scores_A_rec[i, j]:.3f}" for j in range(len(iqa_name))]) + ')')

                links.append(img_path)
            webpage.add_images(ims, txts1, links, additional_txts=txts2)
        webpage.save()

def generate_html_from_json(json_file: str, web_dir='', title='Experienment Results'):
    """Generate an HTML page from a JSON file containing image paths and information."""
    with open(json_file, 'r') as f:
        image_infos = json.load(f)
    if not web_dir:
        web_dir = os.path.dirname(json_file)

    webpage = HTML(web_dir, title, refresh=0)
    for image_info in image_infos:
        # ims, txts, links = [image_info['img_path']], [image_info['label']], [image_info['img_path']]
        header = f"epoch [{image_info['epoch']}]"
        if 'global_step' in image_info and image_info['global_step'] >= 0:
            header += f" (global step {image_info['global_step']})"
        webpage.add_header(f"epoch [{image_info['epoch']}]")
        txt_dict = { k: k for k in image_info['saved_image_name'].keys() }
        if 'original_image_path' in image_info and image_info['original_image_path'] is not None:
            for k in image_info['original_image_path'].keys():
                txt_dict[k] += f" ({image_info['original_image_path'][k]})"
        if 'bpp' in image_info and image_info['bpp'] is not None:
            for k in image_info['bpp'].keys():
                txt_dict[k] += f" (bpp: {image_info['bpp'][k]:.3f})"
        webpage.add_images(
                ims=image_info['saved_image_name'].values(),
                txts=txt_dict.values(),
                links=image_info['saved_image_name'].values())

    webpage.save()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate an HTML page from a JSON file containing image paths and information.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file.')
    parser.add_argument('--web_dir', type=str, default='', help='Directory for the generated HTML page.')
    parser.add_argument('--title', type=str, default='Experiment Results', help='Title for the HTML page.')

    args = parser.parse_args()

    generate_html_from_json(args.json_file, args.web_dir, args.title)

if __name__ == "__main__":
    main()
