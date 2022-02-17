from os import listdir
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from numpy import zeros
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class Guns(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "gun")
        # define data locations
        images_dir = dataset_dir + '/Images/'
        labels_dir = dataset_dir + '/Labels/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-5]
            # skip all images after 266 if we are building the train set
            if is_train and int(image_id) >= 266:
                continue
            # skip all images before 266 if we are building the test/val set
            if not is_train and int(image_id) < 266:
                continue
            img_path = images_dir + filename
            label_path = labels_dir + image_id + '.txt'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=label_path)

    # load all bounding boxes for an image
    def extract_boxes(self, filename):
        boxes = list()
        with open(filename, 'r') as label_file:
            l_count = int(label_file.readline())
            for i in range(l_count):
                box = list(map(int, label_file.readline().split()))
                boxes.append(box)
        return boxes

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load txt
        boxes = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        h = 150
        w = 300
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        #         masks = zeros((*img.shape[:2], 1))
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('gun'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# define a configuration for the model
class GunConfig(Config):
    # define the name of the configuration
    NAME = "gun_cfg"
    # number of classes (background + gun)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 10


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "gun_cfg"
    # number of classes (background + gun)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i * 2 + 2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.show()


# load the train dataset
train_set = Guns()
train_set.load_dataset('./gun', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = Guns()
test_set.load_dataset('./gun', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # plot raw pixel data
    image = train_set.load_image(i)
    pyplot.imshow(image)
    # plot all masks
    mask, _ = train_set.load_mask(i)
    for j in range(mask.shape[2]):
        pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# show the figure
pyplot.show()

config = GunConfig()
# config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('gun_cfg20220213T2112/mask_rcnn_gun_cfg_0005.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)

# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image

    data = pyplot.imread(filename)
     # plot the image
    pyplot.imshow(data)
     # get the context for drawing boxes
    ax = pyplot.gca()
     # plot each box
    for box in boxes_list:
          # get coordinates
        y1, x1, y2, x2 = box
          # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
          # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
        ax.add_patch(rect)
     # show the plot
    pyplot.show()
 

 # create config
cfg = PredictionConfig()
# define the model
rcnn_model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
rcnn_model.load_weights('gun_cfg20220213T2112/mask_rcnn_gun_cfg_0005.h5', by_name=True)
# load photograph
img = load_img('83.jpeg')
img = img_to_array(img)
# make prediction
results = rcnn_model.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes('83.jpeg', results[0]['rois'])
