import tensorflow as tf
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='eg: python3 -img1 file1 -img2 file1 -m 1 -c 0' )
    parser.add_argument('-img1','--image_1',required=True,
                        help='image file_1 URL')
    parser.add_argument('-img2','--image_2',required=True,
                        help='image file_2 URL')
    return parser.parse_args()

def psnr_int(img1,img2):
    # Read images from file.
    im1 = tf.decode_png(img1)
    im2 = tf.decode_png(img2)
    # Compute PSNR over tf.uint8 Tensors.
    psnr1 = tf.image.psnr(im1, im2, max_val=255)

    return psnr1

def psnr_float(img1,img2):
    # Read images from file.
    im1 = tf.decode_png(img1)
    im2 = tf.decode_png(img2)

    # Compute PSNR over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr2 = tf.image.psnr(im1, im2, max_val=1.0)

    return psnr2

def ssim_int(img1,img2):
    # Read images from file.
    im1 = tf.decode_png(img1)
    im2 = tf.decode_png(img2)
    # Compute SSIM over tf.uint8 Tensors.
    ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim1

def ssim_float(img1,img2):
    # Read images from file.
    im1 = tf.decode_png(img1)
    im2 = tf.decode_png(img2)
    # Compute SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    return ssim2

def main():
    args = get_args()
    print(psnr_int(args.image_1,args.image_2))
    print(ssim_int(args.image_1,args.image_2))

if __name__ == '__main__':
    main()