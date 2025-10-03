from visualizer import *


def generate_html_with_selected_images(json_file: str, web_dir: str, title: str = 'Experiment Results', n: int = 5):
    """Generate an HTML page from a JSON file containing image paths and information, selecting every nth image.
    
    Parameters:
        json_file (str): Path to the JSON file containing image information.
        web_dir (str): Directory where the generated HTML page and selected images will be stored.
        title (str): Title for the HTML page.
        n (int): Interval for selecting images. Every nth image will be selected.
    """
    # Load image information from the JSON file
    with open(json_file, 'r') as f:
        image_infos = json.load(f)

    # Create the webpage object
    webpage = HTML(web_dir, title, refresh=0)

    # Ensure the target directories exist
    target_img_dir = os.path.join(web_dir, 'images')
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)

    # Process and add selected images to the webpage
    for i, image_info in enumerate(image_infos):
        if i % n == 0:  # Select every nth image
            header = f"epoch [{image_info['epoch']}]"
            if 'global_step' in image_info and image_info['global_step'] >= 0:
                header += f" (global step {image_info['global_step']})"
            webpage.add_header(header)

            # Prepare text and links for images
            txt_dict = { k: k for k in image_info['saved_image_name'].keys() }
            if 'original_image_path' in image_info and image_info['original_image_path'] is not None:
                for k in image_info['original_image_path'].keys():
                    txt_dict[k] += f" ({image_info['original_image_path'][k]})"
            if 'bpp' in image_info and image_info['bpp'] is not None:
                for k in image_info['bpp'].keys():
                    txt_dict[k] += f" (bpp: {image_info['bpp'][k]:.3f})"

            # Link the images if the web_dir is not the same as the directory containing the JSON file
            ims = []
            links = []
            for original_path in image_info['saved_image_name'].values():
                # Define source and target paths for the image
                src_path = os.path.join(os.path.dirname(json_file), 'images', original_path)
                target_path = os.path.join(target_img_dir, os.path.basename(original_path))
                # Copy the image to the new location
                shutil.copy(src_path, target_path)
                ims.append(os.path.basename(original_path))
                links.append(os.path.basename(original_path))

            webpage.add_images(ims, txt_dict.values(), links)

    # Save the webpage
    webpage.save()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate an HTML page from a JSON file containing image paths and information.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file.')
    parser.add_argument('--web_dir', type=str, required=True, help='Directory for the generated HTML page.')
    parser.add_argument('--title', type=str, default='Experiment Results', help='Title for the HTML page.')
    parser.add_argument('-n', '--n', type=int, default=5, help='Interval for selecting images. Every nth image will be selected.')

    args = parser.parse_args()

    generate_html_with_selected_images(args.json_file, args.web_dir, args.title, args.n)

if __name__ == "__main__":
    main()
