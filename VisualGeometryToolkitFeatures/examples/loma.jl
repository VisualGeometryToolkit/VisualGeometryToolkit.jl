import ONNXRunTime as ORT
using ImageIO
using FileIO
using ImageTransformations
import CUDA, cuDNN
using ImageCore

using VisualGeometryToolkitFeatures

const maxdetections = 2048

descriptor = LoMaBDescriptor()

img_a = imresize(load(download("https://github.com/davnords/LoMa/blob/48cfbec8e56853cdcfd307f97a6b5061c49c4728/assets/toronto_A.jpg?raw=true")), (784, 784))

detections_a =  asift_detect(img_a)
img_a = Float32.(channelview(img_a))
desc_a = descriptor(img_a, detections_a)

img_b = imresize(load(download("https://github.com/davnords/LoMa/blob/48cfbec8e56853cdcfd307f97a6b5061c49c4728/assets/toronto_B.jpg?raw=true")), (784, 784))

detections_b = asift_detect(img_b)
img_b = Float32.(channelview(img_b))

desc_b = descriptor(img_b, detections_b)

matcher = LoMaBMatcher()

matches = matcher(desc_a, desc_b, size(img_a), size(img_b))

const threshold = 0.0

m0 = matches["m0"]
mscores0 = matches["mscores0"]
valid = m0 .> -1 .&& mscores0 .>= threshold
