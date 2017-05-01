# JPEG_like-Compressor
JPEG-like image compressor by Yansen Sheng

jsheng@brandeis.edu

May 1st, 2017

===============================================================

This program is finished independently by Yansen Sheng, but with 
discussion with Ziyu Qiu and Ziqi Wang.

===============================================================

How to use the compressor?

- Simply run compressor.py in terminal. The program is written
  in python2, so make sure python2 is installed.

- The program will first prompt you to either encode or decode.
  To encode, type "encode"; to decode, type "decode". Type "exit"
  to quit the program.

- When encoding, make sure to put the image you want to compress
  in the same folder as the program is in. The program only supports
  .bmp file at current stage. The program will prompt you to enter
  the name of the image file (with suffix). To encode, the program
  will prompt you to choose either to select default block size or
  not. The default block size is 8 by 8. The alternative block size
  is 16 by 16. The program will also prompt you to choose an image
  quality from either low, medium or high. The default image quality
  is low. The program will then compress your chosen image and output
  it with three lossless compress tools, gzip, bzip2 and zip. The
  archive files will be generated in the same folder as the program
  is in.

- When decoding, make sure to put the compressed file you want to
  decompress in the same folder as the program is in. The program
  only supports .gz, .bz2 and .zip files at current stage. The program
  will prompt you to enter the name of the compressed file. Then
  the program will decompress the file and generate a .jpeg image
  file. The program can only generate gray scale image at current
  stage.

===============================================================

What's specific about this compressor?

- This compressor uses JPEG-like encoding algorithm to encode an
  image. The program crops the image into subimage of block size,
  either 8 by 8 or 16 by 16, performs 2-D DCT on the subimage,
  quantizes the subimage with standard quantization matrix and a
  unique factor according to chosen image quality, perform zig-zag
  orientation on the subimage, performs Huffman coding on (the
  difference of) the DC coefficients, and performs Huffman coding
  and run length coding on the AC coefficients.

- For different image quality, the standard quantization matrix is
  divided by a unique factor and round up. Image quality is measured
  by PSNR value which represents the difference between the original
  image and the compressed image. Higher PSNR value means less difference.
  The unique factor is 10 for PSNR = 30, 40 for PSNR = 40, 90 for
  PSNR = 50. The factors are chosen empirically with several tests.
  The factors are the same to block size of 8 and 16.

- DC coefficients and AC coefficients are coded into binary string
  accordingly. DC coefficients use Huffman coding to generate a
  (size, value) pair, where size is a unique index indicating the
  length of value in binary. AC coefficients use Huffman coding to
  generate a (size, value, runlength) tuple, where size is a unique
  index indicating the length of value in binary, and runlength is a
  fixed-length binary indicating the number or consecutive zeros in
  front of this coefficient. Image with block size of 8 uses length
  of 6 and image with block size of 16 uses length of 8. This is not
  the best solution to encode AC tuples, but it works.

- The binary string contains not only DC and AC coefficients, but also
  information of this image. The leading bit represents whether this
  image is compressed with block size of 8 or 16. The following two
  bits indicates whether this image is compressed with low, medium or
  high quality. The following two 8-bit sections represents the width
  and height of the image, accordingly, in the form of actual width/height
  divided by block size. Thus the maximum size of image to compress
  is 2048 * 2048 for block size of 8, and 4096 * 4096 for block size
  of 16.

===============================================================
