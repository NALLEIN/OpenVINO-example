import numpy 
import math
import cv2
import argparse
#python .\PSNR.py -img1 './sr_1.png' -img2 './test1.png'
def get_args():
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='eg: python3 -img1 file1 -img2 file1 -m 1 -c 0' )
    parser.add_argument('-img1','--image_1',required=True,
                        help='image file_1 URL')
    parser.add_argument('-img2','--image_2',required=True,
                        help='image file_2 URL')
    return parser.parse_args()

def main():
    args = get_args()
    im1 = cv2.imread(args.image_1)
    im2 = cv2.imread(args.image_2)
    print(cv2.PSNR(im1,im2))

if __name__ == '__main__':
    main()