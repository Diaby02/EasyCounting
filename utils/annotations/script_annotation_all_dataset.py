import os
import annotator as annotator
import argparse

def main(args):

    dir_path = args.input_dir
    dir_path = os.path.join(dir_path, "images\\")

    files =  [os.path.abspath(os.path.join(dir_path,f)) for f in os.listdir(dir_path)]

    for f in files:
        if ".png" not in f and ".jpg" not in f:
            raise Exception("Your directory does not contain any image")
        else:
            annotator.main(f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input-dir", type=str, default= r"C:\Users\bourezn\Documents\Master_thesis\data\Image_orin\Small_nails2\images_68")

    args = parser.parse_args()
    main(args)

