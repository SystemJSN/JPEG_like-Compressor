from PIL import Image
import numpy as np
import gzip
import bz2
import zipfile

######################################################
#         Setting up global environments             #
######################################################

Q = []                                              # Default quantizaion matrix
Q2 = []                                             # Quantization matrix if block size is 16
C = []                                              # DCT matrix

# Initialize the quatization matrix.
def initializeQMatrix(blockSize, imageQuality):
    global Q
    global Q2
    Q = np.matrix([[16., 11., 10., 16., 24., 40., 51., 61.],
    [12., 12., 14., 19., 26., 58., 60., 55.],
    [14., 13., 16., 24., 40., 57., 69., 56.],
    [14., 17., 22., 29., 51., 87., 80., 62.],
    [18., 22., 37., 56., 68., 109., 103., 77.],
    [24., 35., 55., 64., 81., 104., 113., 92.],
    [49., 64., 78., 87., 103., 121., 120., 101.],
    [72., 92., 95., 98., 112., 100., 103., 99.]])
    Q2 = np.zeros([16, 16])
    if(blockSize == 16):
        Q = upscaleQuantizeMatrix()
    Q = np.ceil(Q / imageQuality)
    return

def upscaleQuantizeMatrix():
    global Q
    global Q2
    i = 7
    while(i >= 0):
        j = 7
        while(j >= 0):
            Q2[2 * i, 2 * j] = Q[i, j]
            Q2[2 * i + 1, 2 * j] = Q[i, j]
            Q2[2 * i, 2 * j + 1] = Q[i, j]
            Q2[2 * i + 1, 2 * j + 1] = Q[i, j]
            j -= 1
        i -= 1
    return Q2

# Initialize the DCT matrix with given block size.
def initializeCMatrix(blockSize):
    global C
    C = np.empty([blockSize, blockSize])
    a = np.sqrt(1. / blockSize)
    for i in range(0, blockSize):
        for j in range(0, blockSize):
            C[i,j] = a * np.cos(np.pi * (2 * j + 1) * i / (2 * blockSize))
        a = np.sqrt(2. / blockSize)
    return

######################################################
#                  Binary dictionary                 #
######################################################

# Dictionary of Huffman coding from integer to binary.
def fromSizeToBinary(size):
    if(size == 0):
        length = '00'
    elif(size == 1):
        length = '010'
    elif(size == 2):
        length = '011'
    elif(size == 3):
        length = '100'
    elif(size == 4):
        length = '101'
    elif(size == 5):
        length = '110'
    elif(size == 6):
        length = '1110'
    elif(size == 7):
        length = '11110'
    elif(size == 8):
        length = '111110'
    elif(size == 9):
        length = '1111110'
    elif(size == 10):
        length = '11111110'
    elif(size == 11):
        length = '111111110'
    elif(size == 12):
        length = '1111111110'
    elif(size == 13):
        length = '11111111110'
    return length

# Dictionary of Huffman coding from binary to integer.
def fromBinaryToSize(length):
    if(length == '00'):
        size = 0
    elif(length == '010'):
        size = 1
    elif(length == '011'):
        size = 2
    elif(length == '100'):
        size = 3
    elif(length == '101'):
        size = 4
    elif(length == '110'):
        size = 5
    elif(length == '1110'):
        size = 6
    elif(length == '11110'):
        size = 7
    elif(length == '111110'):
        size = 8
    elif(length == '1111110'):
        size = 9
    elif(length == '11111110'):
        size = 10
    elif(length == '111111110'):
        size = 11
    elif(length == '1111111110'):
        size = 12
    elif(length == '11111111110'):
        size = 13
    elif(length == '11111111111'):                  # Binary separating AC codes from DC codes.
        size = -1
    else:                                           # Binary not in the dictionary. Continue reading.
        size = -2
    return size

######################################################
#    Functions used by the encoder and decoder       #
######################################################

# Read the image in gray scale.
def readImage(imageName):
    im = Image.open(imageName)
    im = im.convert("L")
    return im

# Applying DCT on the subimage.
def DCT(pixelMatrix, m, n, subImage, blockSize):
    for p in range(0, blockSize):
        for q in range(0, blockSize):
            subImage[p, q] = pixelMatrix[p + m, q + n]
    subImage = np.dot(np.dot(C, subImage), np.transpose(C))
    return subImage

# Applying inverse DCT on the subimage.
# Put the pixels back into the image pixel matrix.
def iDCT(pixelMatrix, m, n, subImage, blockSize):
    subImage = np.dot(np.dot(np.transpose(C), subImage), C)
    for p in range(0, blockSize):
        for q in range(0, blockSize):
            pixelMatrix[p + m, q + n] = subImage[p, q].astype(np.int) + 128
    return

# Applying quantization on the subimage.
def quantize(subImage, blockSize):
    for i in range(0, blockSize):
        for j in range(0, blockSize):
            subImage[i, j] = np.rint(subImage[i, j] / Q[i, j])
    return

# Applying inverse quantization from the subimage
def inverse_quantize(subImage, blockSize):
    for i in range(0, blockSize):
        for j in range(0, blockSize):
            subImage[i, j] = np.rint(subImage[i, j] * Q[i, j])
    return

# Rearrange the subimage in zigzag order.
def zigzag(subImage, blockSize):
    z = np.empty([blockSize*blockSize])
    index = -1
    for i in range(0, 2 * blockSize - 1):
        bound = 0 if i < blockSize else i - blockSize + 1
        for j in range(bound, i - bound + 1):
            index += 1
            z[index] = subImage[j, i-j] if i % 2 == 1 else subImage[i-j, j]
    return z

# Put the DC coefficients and AC coefficients back into the subimage from a zigzag order.
def inverse_zigzag(subImage, DCcode, dcindex, ACcode, blockSize):
    subImage[0, 0] = DCcode[dcindex]
    index = -1
    for i in range(1, len(ACcode)):
        bound = 0 if i < blockSize else i - blockSize + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                subImage[j, i - j] = ACcode[index]
            else:
                subImage[i - j, j] = ACcode[index]
    return

# Get the image size from the compressed file
def getSize(index, fc):
    size = fc[index : index + 8]
    size = int(size, 2)
    return size

# Inverse a binary string bitwise.
def inverseBit(binary):
    binary_inversed = ''
    for bit in binary:
        binary_inversed += '0' if bit == '1' else '1'
    return binary_inversed

# Get the DC coefficients of each subimage.
# DC coefficients coded as the difference of current DC value and the previous DC value.
def DC(subImage_zigzag, subImage_zigzag_prev, DCString):
    dc = subImage_zigzag[0]
    prevdc = subImage_zigzag_prev[0]
    diff = dc - prevdc
    DCString.append(DCcode(diff))
    return

# Turn the DC coefficients into binary string.
def DCcode(diff):
    diff = diff.astype(np.int)
    size = 0
    if(diff != 0):
        diff_abs = np.absolute(diff)
        size = (np.floor(np.log2(diff_abs)) + 1).astype(int)
    length = fromSizeToBinary(size)
    if(diff < 0): diff -= 1
    binary = np.binary_repr(diff, width = size)
    dccode = length + binary[-1 * size:] if diff != 0 else length
    return dccode

# Get the AC coefficients of each subimage.
# AC coefficients arranged in zigzag order.
def AC(subImage_zigzag, ACString, blockSize):
    count0 = 0
    for index in range(1, (blockSize * blockSize)):
        if(subImage_zigzag[index] == 0):
            count0 += 1
        else:
            code = ACcode(subImage_zigzag[index], count0, blockSize)
            ACString.append(code)
            count0 = 0
    if(count0 != 0):
        ACString.append(ACcode(0, count0, blockSize))
    return

# Turn each AC coefficients into binary strings.
def ACcode(ac, count0, blockSize):
    size = 0
    runsize = 6 if blockSize == 8 else 8
    if(ac != 0):
        ac = ac.astype(np.int)
        ac_abs = np.absolute(ac)
        size = (np.floor(np.log2(ac_abs)) + 1).astype(int)
    length = fromSizeToBinary(size)
    if(ac < 0): ac -= 1
    binary = np.binary_repr(ac, width = size)
    run = np.binary_repr(count0, width = runsize)
    accode = length + binary[-1 * size:] + run if ac != 0 else length
    return accode

# Turn binary strings to actual DC coefficients.
def iDC(binary, DCcode, prevDC):
    i = 1
    if(binary[0] == '0'):
        binary = inverseBit(binary)
        i = -1
    code = int(binary, 2) * i
    DCcode.append(prevDC + code)
    return prevDC + code

# Turn binary strings to actual AC coefficients.
def iAC(binary, run, ACcode):
    for x in range(0, run):
        ACcode.append(0)
    i = 1
    if(binary[0] == '0'):
        binary = inverseBit(binary)
        i = -1
    code = int(binary, 2) * i
    ACcode.append(code)
    return

######################################################
#                      Encoder                       #
######################################################
# The encoder takes in an image and compresses it to binary strings
# The compressed file has each components indicating:
#       block size for subimages, in 1 digit;
#       width and height, in 8 digits, respectively;
#       DC coefficients, coded in Huffman style;
#       divider of DC coefficients and AC coefficients, in 9 digits;
#       AC coefficients, coded in runlength and Huffman style.
def encode():
    blockSize = 8               # Default block size for subimage;
    DCString = []               # List storing DC codes. Will convert to string later;
    ACString = []               # List storing AC codes. Will convert to string later;
    divider = '11111111111'     # String separating AC codes from DC codes;
    sizeIndicator = '0'         # String indicating whether the block size is 8 or 16.
                                # Defaul is 0, indicating block size of 8;
    imageQuality = 10           # Expected image quaity for compressed image.
                                # Defalut is low;
    qualityIndicator = '00'     # Strin indicating whether the quality is l, m or h.
                                # Default is 00, indicating l;

    # Choose image.
    imageName = raw_input("Choose Image: ")
    imageName_compressed = imageName[:-4] + '_Compressed'

    # Choose subimage block size.
    # Default is 8.
    chooseBlockSize = 'y'
    chooseBlockSize = raw_input("Do you want to choose the default block size (8*8)? [Y/n]").lower()
    while(chooseBlockSize != 'y' and chooseBlockSize != 'n' and chooseBlockSize!= 'yes' and chooseBlockSize != 'no'):
        chooseBlockSize = raw_input("Please enter either 'y' or 'n' ")
    if chooseBlockSize == 'n' or chooseBlockSize == 'no':
        blockSize = 16
        sizeIndicator = '1'
    print("Your choice of block size is: " + str(blockSize) + "*" + str(blockSize))
    chooseImageQuality = raw_input("Choose image compression quality (l/m/h)").lower()
    if chooseImageQuality == 'm':
        imageQuality = 40
        qualityIndicator = '01'
    elif chooseImageQuality == 'h':
        imageQuality = 90
        qualityIndicator = '10'
    imageName_compressed = imageName_compressed + '_' + str(blockSize) + chooseImageQuality
    initializeQMatrix(blockSize, imageQuality)
    initializeCMatrix(blockSize)

    # Read the image in gray scale and set each pixel in range from -127 to 127.
    # Crop the image to match multiple of the chosen block size.
    image = readImage(imageName)
    pix = image.load()
    width, height = image.size
    width -= (width % blockSize)
    height -= (height % blockSize)
    pixelMatrix = np.zeros([width, height], dtype = int)

    for x in range(0, width):
        for y in range(0, height):
            pixelMatrix[x, y] = pix[x, y] - 128


    # Partition the image into subimage with given size and perform DCT on them.
    subImage_zigzag_prev = np.zeros([blockSize * blockSize])
    m = 0
    while(m < width):
        n = 0
        while(n < height):
            subImage = np.empty([blockSize, blockSize])
            subImage = DCT(pixelMatrix, m, n, subImage, blockSize)
            quantize(subImage, blockSize)
            subImage_zigzag = zigzag(subImage, blockSize)
            DC(subImage_zigzag, subImage_zigzag_prev, DCString)
            AC(subImage_zigzag, ACString, blockSize)
            subImage_zigzag_prev = subImage_zigzag
            n += blockSize
        m += blockSize

    # Storing DC coefficients and AC coefficients into strings.
    DCString = ''.join(DCString)
    ACString = ''.join(ACString)

    # String indicating the size of the image being compressed.
    # The size is represented by a 8-digit binary bumber.
    # Maximun size is 2040 * 2040 for block size of 8, and 4080 * 4080 for block size of 16.
    widthSize = np.binary_repr(width / blockSize, width = 8)
    heightSize = np.binary_repr(height / blockSize, width = 8)

    # The compressed file has each components indicating:
    #       block size for subimages, in 1 digit;
    #       width and height, in 8 digits, respectively;
    #       DC coefficients, coded in Huffman style;
    #       divider of DC coefficients and AC coefficients, in 9 digits;
    #       AC coefficients, coded in runlength and Huffman style.

    # Compress the data.
    compressionData = sizeIndicator + qualityIndicator + widthSize + heightSize + DCString + divider + ACString
    # Compress with gzip.
    output = gzip.open(imageName_compressed + '.txt.gz', 'wb')
    try:
        output.write(compressionData)
    finally:
        output.close()
    # Compress with bzip2.
    output = bz2.BZ2File(imageName_compressed + '.txt.bz2', 'wb')
    try:
        output.write(compressionData)
    finally:
        output.close()
    # Compress with zip
    output = zipfile.ZipFile(imageName_compressed + '.txt.zip',
                     mode='w',
                     compression=zipfile.ZIP_DEFLATED,
                     )
    try:
        output.writestr(imageName_compressed + '.txt', compressionData)
    finally:
        output.close()


    print('Your compressed image is stored as ' + imageName_compressed + '.txt in three different compressed formats.')


    return


######################################################
#                     Decoder                        #
######################################################
# The decoder takes in a binary string and decompress it to a imgae file.
# Different from how it was done while encoding, DC coefficients and AC coefficients are stored separately.
# Thus the decoder first read all the DC coefficients and store them in a list, then read the AC coefficients.
# The decoder keeps track of how many AC coefficients has been read (63 if the block size is 8 and 255 if the
# block size is 16) and store them into the image matrix with the corresponding DC coefficients.
def decode():
    blockSize = 8               # Block size for subimages.
                                # Default is 8;
    runlength = 6               # Size of runlength code.
                                # Default is 6;
    divider = 0                 # Flag for divider.
    length = ''                 # Binary string of Huffman length
    binary = ''                 # Binary string of binary number code
    run = ''                    # Binary string of runlength code
    DCcode = []                 # List to store decoded DC coefficients
    ACcode = []                 # List to store decoded AC coefficients
    imageQuality = 10           # Expected image quaity for compressed image.
                                # Defalut is low;


    # Choose the image file to decompress.
    imageName = raw_input("Choose File To Decompress: ")
    print("The file is: " + imageName)

    # Read the file according to the suffix of the file.
    if imageName.endswith('.zip'):
        input_file = zipfile.ZipFile(imageName)
        imageName = imageName[:-4]
        fc = input_file.read(imageName)
    else:
        if imageName.endswith('.gz'):
            input_file = gzip.open(imageName, 'rb')
            imageName = imageName[:-3]
        elif imageName.endswith('.bz2'):
            input_file = bz2.BZ2File(imageName, 'rb')
            imageName = imageName[:-4]
        fc = input_file.read()
    input_file.close()

    # Get block size of subimage.
    index = 0
    if(fc[0] == '1'):
        blockSize = 16
        runlength = 8

    if(fc[1:3] == '01' and blockSize == 8):
        imageQuality = 40
    elif(fc[1:3] == '10'and blockSize == 8):
        imageQuality = 90
    elif(fc[1:3] == '00'):
        imageQuality = 10
    elif(fc[1:3] == '01'):
        imageQuality = 40
    elif(fc[1:3] == '10'):
        imageQuality = 90

    # Get image size.
    index = 3
    width = getSize(index, fc) * blockSize
    index += 8
    height = getSize(index, fc) * blockSize
    index += 8

    # Initialize quantization matrix and DCT matrix.
    initializeQMatrix(blockSize, imageQuality)
    initializeCMatrix(blockSize)

    # Initialize image matrix.
    pixelMatrix = np.zeros([width, height])

    # Decode DC coefficients and store them in a list.
    # Binary codes for DC coefficients are read from the file bitwise until the binary is found in the dictionary.
    prevDC = 0
    while(index < len(fc) and divider == 0):
        length += fc[index]
        size = fromBinaryToSize(length)
        if(size == -1):
            divider = 1
            index += 1
            length = ''
        elif(size == -2):
            index += 1
        else:
            index += 1
            if(size == 0):
                DCcode.append(prevDC + 0)
            else:
                binary = fc[index : index + size]
                index += size
                prevDC = iDC(binary, DCcode, prevDC)
            length = ''

    # Decode AC coefficients for each subimage and store them into the image matrix with teh DC coefficients.
    # Binary codes for AC coefficients are read from the file bitwise until the binary is found in the dictionary.
    dcindex = 0                             # Index to keep track to get the correct DC coefficient.
    m = 0                                   # Width index of the subimage.
    n = 0                                   # Height index of the subimage.
    count = blockSize * blockSize - 1       # Counter to keep track to get correct number of AC coefficients.
                                            # Counter is 63 if block size is 8 and is 255 if block size is 16.
    while(index < len(fc)):
        length += fc[index]
        size = fromBinaryToSize(length)
        if(size == -2):
            index += 1
        else:
            index += 1
            if(size == 0):
                while count > 0:
                    ACcode.append(0)
                    count -= 1
            else:
                binary = fc[index : index + size]
                index += size
                run = fc[index : index + runlength]
                run = int(run, 2)
                index += runlength
                iAC(binary, run, ACcode)
                count = count - run - 1
            length = ''

            if(count == 0):                 # The end of a subimage
                subImage = np.zeros([blockSize, blockSize])
                inverse_zigzag(subImage, DCcode, dcindex, ACcode, blockSize)
                inverse_quantize(subImage, blockSize)
                # if(m == 0 and n == 0):
                #     print("After iQuantization")
                #     print(subImage)
                iDCT(pixelMatrix, m, n, subImage, blockSize)
                dcindex += 1
                n = (n + blockSize) % height
                if(n == 0): m += blockSize
                ACcode = []
                count = blockSize * blockSize - 1

    # Store the image in jpeg file and show the image.
    new_im = Image.fromarray(np.uint8(np.transpose(pixelMatrix)))
    new_im.save(imageName[:-4] + '.jpeg', "jpeg")

# Main loop.
loop = 1
while(loop == 1):
    prompt = raw_input('Enter encode or decode to proceed; enter exit to quit: ').lower()
    if(prompt == 'encode'):
        encode()
    elif(prompt == 'decode'):
        decode()
    elif(prompt == 'exit'):
        loop = 0
    else:
        print('input not accepted')
