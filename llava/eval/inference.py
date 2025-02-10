import torch, os, sys
import torch
import argparse
import llava
from loguru import logger
from llava import conversation as clib
from llava.media import Image

from termcolor import colored


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")  
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")

    args = parser.parse_args()

    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
    breakpoint()
    model = llava.load(args.model_path)

    logger.info(f"Model loaded from path: {args.model_path}")

    prompt = []
    if args.media is not None:
        for media in args.media or []:
            media = Image(media)
            # img = Image.open(image_fpath).convert("RGB")

            # crop_size = self.data_args.image_processor.size
            # img = img.resize((crop_size["width"], crop_size["height"]))
        
        prompt.append(media)

    if args.text is not None:
        prompt.append(args.text)


    response = model.generate_content(prompt)
    print(f"\n\n {os.path.basename(args.model_path)} RESPONSE \n {colored(response, 'cyan', attrs=['bold'])}")

if __name__ == "__main__":
    main()