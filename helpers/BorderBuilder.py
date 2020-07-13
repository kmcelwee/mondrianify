from PIL import Image
import cv2
import os

class BorderBuilder:
    """BorderBuilder is a class that helps apply Holisticly-Nested Edge Detection
    to an image so that we can get the major features of an image.
    """
    def __init__(
        self, 
        image_in,
        prototxt=os.path.dirname(__file__) + "/hed_model/deploy.prototxt",
        caffemodel=os.path.dirname(__file__) + "/hed_model/hed_pretrained_bsds.caffemodel",
        hed_threshold=190
    ):
        self.image_in = image_in
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.hed_threshold = hed_threshold

        # Vars to be set later
        self.hed = None
        self.pos_ids = None

    def apply_hed(self):
        """Apply HED to the input image"""
        # Load neural network
        net = cv2.dnn.readNetFromCaffe(
            self.prototxt, 
            self.caffemodel
        )

        image = cv2.imread(self.image_in)
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False
        )

        # Apply neural network and store output image
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")

        self.hed = hed

    def save_hed(self, file):
        """Save HED's output image to the given file"""
        cv2.imwrite(file, self.hed)


    def apply_hed_threshold(self):
        """Apply a cutoff so that all values in the array are minmaxed 
        into a binary.
        """
        n_array = self.hed
        n_array[n_array < self.hed_threshold] = 0
        n_array[n_array >= self.hed_threshold] = 250

        self.pos_ids = n_array


    def save_threshold(self, file):
        """Save the threshold image into the given file"""
        im2 = Image.fromarray(self.pos_ids)
        im2.save(file)


class CropLayer(object):
    """A class helper to ensure that the HED cropper is properly configured"""
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

# HACK: We need to add this layer, but it can only be done once, and it seems
#   like cv2 keeps settings from previous runs to save on memory? And there's no 
#   straightforward way to ensure that this crop layer was added. Loading it once 
#   outside the class seems to work best.
cv2.dnn_registerLayer("Crop", CropLayer)
