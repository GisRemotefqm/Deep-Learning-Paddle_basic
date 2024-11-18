import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout
import random
import os
import xml.etree.ElementTree as ET
import cv2 as cv
from PIL import Image, ImageEnhance

"""
size: 图片尺寸
object：图片中包含的物体，一张图片可能包含多个物体
name：昆虫名称
bndbox：物体真实框
difficult：识别是否困难

"""
INSECT_NAMES = ['Boerner', 'Leconte',
                'Linnaeus', 'acuminatus',
                'armandi', 'coleoptera',
                'linnaeus']


def get_insect_name():
    name_to_dict = {}

    for i, name in enumerate(INSECT_NAMES):
        name_to_dict[name] = i

    return name_to_dict


# 获取xml文件中的标注信息
def get_xml_information(cname2cid, datadir):
    filenames = os.listdir((os.path.join(datadir, 'annotations', 'xmls')))
    records = []
    ct = 0
    for name in filenames:

        fid = name.split('.')[0]
        xml_path = os.path.join(datadir, 'annotations', 'xmls', name)
        img_path = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(xml_path)

        if tree.find('id') is None:
            img_id = np.array([ct])
        else:
            img_id = np.array([int(tree.find('id').text)])

        objt = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_box = np.zeros((len(objt), 4), dtype='float32')
        gt_class = np.zeros((len(objt), ), dtype='int32')
        is_crowd = np.zeros((len(objt), ), dtype=np.int32)
        difficult = np.zeros((len(objt), ), dtype=np.int32)

        for i, obj in enumerate(objt):

            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = obj.find('difficult').text
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = np.maximum(x1, 0)
            y1 = np.maximum(y1, 0)
            x2 = np.minimum(im_w - 1, x2)
            y2 = np.minimum(im_h - 1, y2)

            # 使用x, y, w, h格式表示真实框位置
            gt_box[i] = [(x1 + x2) / 2., (y1 + y2) / 2., x2 - x1 + 1., y2 - y1 + 1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'img_path': img_path,
            'img_id': img_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_box': gt_box,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objt) != 0:
            records.append(voc_rec)
        ct += 1

    return records


def get_box(box, cls):
    """
    一张图片上往往有多个目标，设置参数MAX_NUM即一张图片最多有50个真实框
    若真实框数目小于50，则不足的gt_box，gt_class和gt_score的各项数值设为0
    :return:
    """
    MAX_NUM = 50
    gt_box = np.zeros((MAX_NUM, 4), dtype=np.float32)
    gt_class = np.zeros((MAX_NUM, ), dtype=np.int32)

    for i in range(len(box)):
        gt_box[i] = box[i]
        gt_class[i] = cls[i]

        if i >= MAX_NUM:
            break
    return gt_box, gt_class


def get_imgdata(record):
    img_path = record['img_path']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_box']
    difficult = record['difficult']
    print(img_path)
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 检查在xml文件中读取的h和w与img的h，w是否相等
    assert h == img.shape[0], 'image height of {} inconsistent in record({}) and img file({})'.format(img_path, img.shape[0])
    assert w == img.shape[1], 'image width of {} inconsistent in record({}) and img file({})'.format(img_path, img.shape[1])

    gt_box, gt_labes = get_box(gt_bbox, gt_class)

    # gt_box中的值用相对值
    gt_box[:, 0] = gt_box[:, 0] / float(w)
    gt_box[:, 1] = gt_box[:, 1] / float(h)
    gt_box[:, 2] = gt_box[:, 2] / float(w)
    gt_box[:, 3] = gt_box[:, 3] / float(h)

    return img, gt_box, gt_labes, (h, w)


# 随机改变亮度，对比度颜色
def radom_distort(img):

    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)

        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)

        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)

        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img

# 随机填充
def random_expand(img, gtboxes, max_ratio=4, fill=None, keep_ratio=True, threah=0.5):

    if random.random() > threah:
        return img, gtboxes

    if max_ratio < 1:
        return img, gtboxes

    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c))

    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes


# 随机裁剪
def muli_box_iou_xy(box1, box2):
    """
    在random_clip中使用
    :param box1:
    :param box2:
    :return:
    """

    assert box1.shape[-1] == 4, 'box1 shape[-1] should be 4'
    assert box2.shape[-1] == 4, 'box2.shape[-2] should be 4'

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2,\
                   box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2,\
                   box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2,\
                   box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2,\
                   box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0, a_max=None)
    inter_h = np.clip(inter_h, a_min=0, a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    """
    在random_clip中使用
    :param boxes:
    :param labels:
    :param crop:
    :param img_shape:
    :return:
    """
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)
    boxes = boxes.copy()

    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w,\
                               (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h,\
                               (boxes[:, 1] + boxes[:, 3] / 2) * im_h
    crop_box = [x, y, x + w, y + h]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')

    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w,\
                               (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h,\
                               (boxes[:, 1] - boxes[:, 3]) / h

    return boxes, labels, mask.sum()


# 随机裁剪
def random_clip(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constrain=None, max_trial=50):

    if len(boxes) == 0:
        return img, boxes

    if not constrain:
        constrain = [(0.1, 1.0), (0.3, 1.0),
                     (0.5, 1.0), (0.7, 1.0),
                     (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constrain:
        for trial in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_iou, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[(crop_x + crop_w / 2.0) / w,
                                 (crop_y + crop_h / 2.0) / h,
                                 crop_w / float(w),
                                 crop_h / float(h)]])

            iou = muli_box_iou_xy(crop_box, boxes)

            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_box, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue

        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_box, crop_labels

    img.asarray(img)
    return img, boxes, labels


# 随机缩放
def random_interp(img, size, interp=None):

    interp_method = [
        cv.INTER_NEAREST,
        cv.INTER_LINEAR,
        cv.INTER_AREA,
        cv.INTER_CUBIC,
        cv.INTER_LANCZOS4,
    ]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]

    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)

    return img


# 随机翻转
def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1. - gtboxes[:, 0]

    return img, gtboxes


# 随机打乱真实框排序
def shuffle_gtbox(gtbox, gtlabel):
    gt = np.concatenate([gtbox, gtlabel[:, np.newaxis]], axis=1)
    idx = np.arange(gt.shape[0])
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]


# 图像增广方法汇总
def img_augment(img, gtboxes, gtlabels, size, mean=None):

    # 随机改变明暗、对比度
    img = radom_distort(img)

    # 随机填充
    img, gtboxes = random_expand(img, gtboxes, fill=mean)

    # 随机裁剪
    img, gtboxes, gtlabels = random_clip(img, gtboxes, gtlabels)

    # 随机缩放
    img = random_interp(img, size)

    # 随机翻转
    img, gtboxes = random_flip(img, gtboxes)

    # 随机打乱真实框排序
    gtboxes, gtlabels = shuffle_gtbox(gtboxes, gtlabels)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')


def get_img_data(record, size=640):

    img, gt_boxes, gt_labels, scales = get_imgdata(record)
    print(img.shape)
    img, gt_boxes, gt_labels = img_augment(img, gt_boxes, gt_labels, size)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))

    return img, gt_boxes, gt_labels, scales


class TrainDataset(paddle.io.Dataset):

    def __init__(self, datadir, mode='train'):

        self.datadir = datadir
        cname2cid = get_insect_name()
        self.records = get_xml_information(cname2cid, datadir)
        self.img_size = 640

    def __getitem__(self, item):

        record = self.records[item]
        img, gt_boxes, gt_labels, img_shape = get_img_data(record, size=self.img_size)

        return img, gt_boxes, gt_labels, np.array(img_shape)

    def __len__(self):

        return len(self.records)


# 将list形式的batch数据转化成多个array构成tuple
def make_test_array(batch_data):

    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype='float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')

    return img_name_array, img_data_array, img_scale_array


def test_data_loader(datadir, batch_size=10, test_image_size=608, mode='test'):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    :param datadir:
    :param batch_size:
    :param test_image_size:
    :param mode:
    :return:
    """
    img_names = os.listdir(datadir)
    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in img_names:
            file_path = os.path.join(datadir, image_name)
            img = cv.imread(file_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv.resize(img, (img_size, img_size))

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) /std
            out_img = out_img.astype('float32').transpose((2, 0, 1))
            img = out_img
            img_shape = [H, W]

            batch_data.append((image_name.split('.')[0], img, img_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)

        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader


"""
    YOLOv3模型思想
        按照一定规则在图片上产生一系列的候选区域，然后根据这些候选区与图片上真实框之间的位置
    关系对候选区域进行标注。跟真实框足够接近的那些候选区会被标注为正样本，
    同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区则会标注为负样本
    ，负样本不需要预测位置喝类别。
        使用卷积神经网络提取图片特征并对候选区域的位置和类别进行预测。这样每个预测框
    就可以看成一个样本，根据真实框相对它的位置和类别进行了标注而活得标签值，通过
    网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立损失函数。
        
"""


# 标注预测框的objectness
def get_objectnesss_label(img, gt_boxes, gt_labels, iou_threshold=0.7,
                          anchors=[116, 90, 156, 198, 373, 326],
                          num_classes=7, downsample=32):
    """

    :param img:图像数据，形状[N,C,H,W]
    :param gt_boxes: 真实框[N,50,4],使用的相对值
    :param gt_labels: [N,50]真实框类别
    :param iou_threshold: 当预测框的iou大于iou_threshold时不将其看作负样本
    :param anchors: 锚框尺寸
    :param num_classes: 类别数目
    :param downsample: 特征图相对于输入网络的图片尺寸变化比例
    :return:
    """

    img_shape = img.shape
    batchsize = img_shape[0]
    num_anchors = len(anchors) // 2
    input_h = img_shape[2]
    input_w = img_shape[3]

    num_rows = input_h / downsample
    num_cols = input_w / downsample

    label_objectness = np.zeros([batchsize, num_anchors, num_rows, num_cols])
    label_classification = np.zeros([batchsize, num_anchors, num_classes, num_rows, num_cols])
    label_location = np.zeros([batchsize, num_anchors, 4, num_rows, num_cols])

    scale_location = np.ones([batchsize, num_anchors, num_rows, num_cols])

    # 对batchsize循环，处理每张图片
    for n in range(batchsize):
        # 对图片上的真实框循环，依次找出跟真实框形状最匹配的锚框
        for n_gt in range(len(gt_boxes[n])):

            gt_box = gt_boxes[n][n_gt]
            gt_cls = gt_labels[n][n_gt]
            gt_center_x = gt_box[0]
            gt_center_y = gt_box[1]
            gt_width = gt_box[2]
            gt_height = gt_box[3]
            if (gt_width < 1e-3) or (gt_height < 1e-3):
                continue
            i = int(gt_center_y * num_rows)
            j = int(gt_center_x * num_cols)
            ious = []

            for ka in range(num_anchors):
                bbox1 = [0, 0, float(gt_width), float(gt_height)]
                anchor_w = anchors[ka * 2]
                anchor_h = anchors[ka * 2 + 1]
                bbox2 = [0, 0, anchor_w/float(input_w), anchor_h/float(input_h)]

                # 计算iou
                iou = box_iou_xywh(bbox1, bbox2)
                ious.append(iou)

            ious = np.array(ious)
            inds = np.argsort(ious)
            k = inds[-1]
            label_objectness[n, k, i, j] = 1
            c = gt_cls
            label_classification[n, k, c, i, j] = 1.

            dx_label = gt_center_x * num_cols - j
            dy_label = gt_center_y * num_rows - i
            dw_label = np.log(gt_width * input_w / anchors[k * 2])
            dh_label = np.log(gt_height * input_h / anchors[k * 2 +1])
            label_location[n, k, 0, i, j] = dx_label
            label_location[n, k, 1, i, j] = dy_label
            label_location[n, k, 2, i, j] = dw_label
            label_location[n, k, 3, i, j] = dh_label
            scale_location[n, k, i, j] = 2.0 - gt_width * gt_height

    return label_objectness.astype('float32'), label_classification.astype('float32'), scale_location.astype('float32')


# 计算iou
def box_iou_xywh(box1, box2):

    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box2[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)

    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_w * inter_h

    union = s1 + s2 - intersection
    iou = intersection / union

    return iou


class ConvBNLayer(paddle.nn.Layer):
    
    def __init__(self, ch_in, ch_out,
                 kernel_size=3, stride=1,
                 groups=1, padding=0,
                 act='leaky'):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(in_channels=ch_in,
                           out_channels=ch_out,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           weight_attr=paddle.ParamAttr(
                               initializer=paddle.nn.initializer.Normal(0, 0.02),
                           ),
                           bias_attr=False)

        self.batch_norm = paddle.nn.BatchNorm2D(num_features=ch_out,
                                                weight_attr=paddle.ParamAttr(
                                                    initializer=paddle.nn.initializer.Normal(0, 0.02),
                                                    regularizer=paddle.regularizer.L2Decay(0.)
                                                ),
                                                bias_attr=paddle.ParamAttr(
                                                    initializer=paddle.nn.initializer.Constant(0.0),
                                                    regularizer=paddle.regularizer.L2Decay(0.0)))

        self.act = act

    def forward(self,inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)

        if self.act == 'leaky':
            out = F.leaky_relu(x=out, negative_slope=0.1)

        return out


class DownSample(paddle.nn.Layer):

    def __init__(self, ch_in, ch_out, kernel_size=3, stride=2, padding=1):
        super(DownSample, self).__init__()
        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in,
                                        ch_out=ch_out,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
        self.ch_out = ch_out

    def forward(self, inputs):

        out = self.conv_bn_layer(inputs)

        return out


class BasicBlock(paddle.nn.Layer):
    """
    定义残差块，输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
    """

    def __init__(self, ch_in, ch_out):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(ch_in=ch_in,
                                 ch_out=ch_out,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.conv2 = ConvBNLayer(ch_in=ch_out,
                                 ch_out=ch_out * 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

    def forward(self, inputs):

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.add(x=inputs, y=conv2)

        return out


class LayerWarp(paddle.nn.Layer):
    
    """
    添加多层残差块，组成Darknet53网络的一个层级
    """
    def __init__(self, ch_in, ch_out, count, is_test=True):
        super(LayerWarp, self).__init__()

        self.basicblock0 = BasicBlock(ch_in, ch_out)
        self.res_out_list = []

        for i in range(1, count):

            # 用来添加子层
            res_out = self.add_sublayer('basic_block_%d' % i,
                                        BasicBlock(ch_out * 2, ch_out))
            self.res_out_list.append(res_out)

    def forward(self, inputs):

        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)

        return y


# 每个残差块的个数，来自DarkNet的网络结构图
DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(paddle.nn.Layer):

    def __init__(self):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(ch_in=3,
                                 ch_out=32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        # 下采样，使用stride=2的卷积实现
        self.downsample0 = DownSample(ch_in=32,
                                      ch_out=32 * 2)

        # 添加各个层级
        self.darknet53_conv_block_list = []
        self.downsample_list = []

        for i, stage in enumerate(self.stages):

            conv_block = self.add_sublayer(
                'stage_%d' % i,
                LayerWarp(32 * (2 ** (i + 1)),
                          32 * (2 ** i),
                          stage)
            )
            self.darknet53_conv_block_list.append(conv_block)

        for i in range(len(self.stages) - 1):

            downsample = self.add_sublayer(
                'stage_%d_downsample' % i,
                DownSample(ch_in=32 * (2 ** (i + 1)),
                           ch_out=32 * (2 ** (i + 2))))

            self.downsample_list(downsample)


    def forward(self, inputs):

        out = self.conv0(inputs)
        out = self.downsample0(out)

        block = []

        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):

            out = conv_block_i(out)
            block.append(out)

            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)

            return block[-1: -4: -1]


class YoloDetectionBlock(paddle.nn.Layer):

    def __init__(self, ch_in, ch_out, is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, 'channel {} cannot be divided by 2'.format(ch_out)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv1 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = ConvBNLayer(
            ch_in=ch_out * 2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv3 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.route = ConvBNLayer(
            ch_in= ch_out * 2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out= ch_out * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)

        return route, tip


def sigmoid(x):

    return 1. / (1.0 + np.exp(-x))


# 将网络特征图输出的[tx, ty, th, tw]转化成预测框的坐标[x1, y1, x2, y2]
def get_yolo_box_xy(pred, anchors, num_classes, downsample):

    """
    :param pred: 网络输出特征图转化成的numpy.ndarray
    :param anchors: list，表示锚框的大小
    :param num_classes:
    :param downsample:
    :return:
    """

    batchsize = pred.shape[0]
    num_rows = pred.shape[-2]
    num_cols = pred.shape[-1]

    input_h = num_rows * downsample
    input_w = num_cols * downsample
    num_anchors = len(anchors) // 2

    # pred的形状是[N, C, H, W]其中C=NUM_ANCHORS * (5 + NUM_CLASSES)

    pred = pred.reshape([-1, num_anchors, 5 + num_classes, num_rows, num_cols])
    pred_location = pred[:, :, 0:4, :, :]
    pred_location = np.transpose(pred_location, (0, 3, 4, 1, 2))
    anchors_this = []

    for ind in range(num_anchors):

        anchors_this.append([anchors[ind * 2], anchors[ind * 2 + 1]])

    anchors_this = np.array(anchors_this).astype('float32')

    pred_box = np.zeros(pred_location.shape)

    for n in range(batchsize):
        for i in range(num_rows):
            for j in range(num_cols):
                for k in range(num_anchors):

                    pred_box[n, i, j, 0] = j
                    pred_box[n, i, j, k, 1] = i
                    pred_box[n, i, j, k ,2] = anchors_this[k][0]
                    pred_box[n, i, j, k, 3] = anchors_this[k][1]

    # 使用相对坐标，pred_box的输出元素数值在0~1之间
    pred_box[:, :, :, :, 0] = (sigmoid(pred_location[:, :, :, :, 0]) + pred_box[:, :, :, :, 0]) / num_cols
    pred_box[:, :, :, :, 1] = (sigmoid(pred_location[:, :, :, :, 1]) + pred_box[:, :, :, :, 1]) / num_rows
    pred_box[:, :, :, :, 2] = (sigmoid(pred_location[:, :, :, :, 2]) + pred_box[:, :, :, :, 2]) / input_w
    pred_box[:, :, :, :, 3] = (sigmoid(pred_location[:, :, :, :, 3]) + pred_box[:, :, :, :, 3]) / input_h

    # 将坐标从xywh转化为xyxy
    pred_box[:, :, :, :, 0] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2] / 2.
    pred_box[:, :, :, :, 1] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3] / 2.
    pred_box[:, :, :, :, 2] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2]
    pred_box[:, :, :, :, 3] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3]

    pred_box = np.clip(pred_box, 0, 1.0)

    return pred_box


# 挑选出和真实框IOU大于阈值的预测框
def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_thireshold):

    batchsize = pred_box.shape[0]
    num_rows = pred_box.shape[1]
    num_cols = pred_box.shape[2]
    num_anchors = pred_box.shape[3]
    ret_inds = np.zeros([batchsize, num_rows, num_cols, num_anchors])

    for i in range(batchsize):

        pred_box_i = pred_box[i]
        gt_boxes_i = gt_boxes[i]

        for k in range(len(gt_boxes_i)):

            gt = gt_boxes_i[k]
            gtx_min = gt[0] - gt[2] / 2.
            gty_min = gt[1] - gt[3] / 2.
            gtx_max = gt[0] + gt[2] / 2.
            gty_max = gt[1] - gt[3] / 2.

            if (gtx_max - gtx_min < 1e-3) or (gty_max - gty_min < 1e-3):
                continue

            x1 = np.maximum(pred_box_i[:, :, :, :, 0], gtx_min)
            y1 = np.maximum(pred_box_i[:, :, :, :, 1], gty_min)
            x2 = np.minimum(pred_box_i[:, :, :, :, 2], gtx_max)
            y2 = np.minimum(pred_box_i[:, :, :, :, 3], gty_max)

            intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)

            s1 = (gty_max - gty_min) * (gtx_max - gtx_min)
            s2 = (pred_box_i[:, :, :, :, 2] - pred_box_i[:, :, :, :, 0]) * (pred_box_i[:, :, :, :, 3] - pred_box_i[:, :, :, :, 1])

            union = s2 + s1 -intersection
            iou = intersection / union
            above_inds = np.where(iou > iou_thireshold)
            ret_inds[i][above_inds] = 1

    ret_inds = np.transpose(ret_inds, (0, 3, 1, 2))

    return ret_inds.astype('bool')


# 计算损失函数
def get_loss(output, label_objectness, label_location, label_classification, scales, num_anchors=3, num_classes=7):

    # 将output从[N, C, H, W]变形为[N, NUM_ANCHORs, NUM_CLASSES + 5, H, W]
    reshape_output = paddle.reshape(output, [-1, num_anchors, num_classes + 5,output.shape[2], output.shape[3]])

    # 从output中取出和objectness相关的预测值
    pred_objectness = reshape_output[:, :, 4, :, :]
    loss_objectness = F.binary_cross_entropy_with_logits(pred_objectness, label_objectness, reduction='none')

    # pos_sample只有在正样本的地方取值为1，其他地方取值全为0
    pos_objectness = label_objectness > 0
    pos_samples = paddle.cast(pos_objectness, 'float32')
    pos_samples.stop_gradient = True

    # 从output中取出所有跟位置相关的预测值
    tx = reshape_output[:, :, 0, :, :]
    ty = reshape_output[:, :, 1, :, :]
    tw = reshape_output[:, :, 2, :, :]
    th = reshape_output[:, :, 3, :, :]

    # 从label中取出各个位置坐标的标签
    dx_label = label_location[:, :, 0, :, :]
    dy_label = label_location[:, :, 1, :, :]
    tw_label = label_location[:, :, 2, :, :]
    th_label = label_location[:, :, 3, :, :]

    # 计算各个位置的损失函数
    loss_location_x = F.binary_cross_entropy_with_logits(tx, dx_label, reduction='none')
    loss_location_y = F.binary_cross_entropy_with_logits(ty, dy_label, reduction='none')
    loss_location_w = paddle.abs(tw - tw_label)
    loss_location_h = paddle.abs(th - th_label)

    # 计算总的损失函数
    loss_location = loss_location_x + loss_location_y + loss_location_w + loss_location_h
    # 乘以scales
    loss_location = loss_location * scales
    # 只计算正样本位置的损失函数
    loss_location = loss_location * pos_samples

    # 从output中取出所有跟物体类别相关的像素点
    pred_classification = reshape_output[:, :, 5: num_classes + 5, :, :]

    # 计算分类相关的损失函数
    loss_classification = F.binary_cross_entropy_with_logits(pred_classification, label_classification, reduction='none')

    # 将第2维求和
    loss_classification = paddle.sum(loss_classification, axis=2)
    # 只计算objectness为正的样本的分类损失函数
    loss_classification = loss_classification * pos_samples
    total_loss = loss_objectness + loss_location + loss_classification
    # 对所有有预测框的loss进行求和
    total_loss = paddle.sum(total_loss, axis=[1, 2, 3])
    # 对所有样本求平均
    total_loss = paddle.mean(total_loss)

    return total_loss


# 多尺度检测
# 因为使用多尺度检测，所以可能要进行大幅度修改使用paddle.vision.ops.yolo_loss

# 定义采样模块
class Upsample(paddle.nn.Layer):

    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):

        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisize by actual_shape
        out = paddle.nn.functional.interpolate(
            x=inputs,
            scale_factor=self.scale,
            mode='NEAREST'
        )
        return out


class YOLOv3(paddle.nn.Layer):

    def __init__(self, num_classes=7):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        # 提取图像特征的骨干代码
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        # 生成3个层级的特征图P0, P1, P2
        for i in range(3):
            yolo_block = self.add_sublayer(
                'yolo_detection_block_%d' % i,
                YoloDetectionBlock(
                    ch_in=512 // (2 ** i) * 2 if i == 0 else 512 // (2 ** i) * 2 + 512 // (2 ** i),
                    ch_out=512 // (2 ** i)
                )
            )
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (self.num_classes + 5)

            # 添加从ti生成pi的模块，这是个Conv2D操作，输出通道数为3 * (num_classes + 5)
            block_out = self.add_sublayer(
                'block_out_%d' % i,
                Conv2D(in_channels= 512 // (2 ** i) * 2,
                       out_channels=num_filters,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       weight_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Normal(0, 0.02)
                       ),
                       bias_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Constant(0, 0),
                           regularizer=paddle.regularizer.L2Decay(0.)
                       ))
            )

            self.block_outputs.append(block_out)
            if i < 2:
                # 对ri进行卷积
                route = self.add_sublayer(
                    'route2_%d' % i,
                    ConvBNLayer(ch_in=512 // (2 ** i),
                                ch_out=256 // (2 ** i),
                                kernel_size=1,
                                stride=1,
                                padding=0)
                )
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_(i + 1)保持同样的尺寸
            self.upsample = Upsample()

    def forward(self, inputs):

        outputs = []
        blocks = self.block(inputs)

        for i, block in enumerate(blocks):

            if i > 0:

                # 将r_(i -1)经过卷积和上采样之后得到特征图,与这一级的ci进行拼接
                block = paddle.concat([route, block], axis=1)

            # 从ci生成ti和ri
            route, tip = self.yolo_blocks[i](block)
            # 从ti生成pi
            block_out = self.block_outputs[i](tip)
            # 将pi放入列表
            outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_(i + 1)保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(self,
                 outputs,
                 gtbox,
                 gtlabel,
                 gtscore=None,
                 anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchors_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_thresh=0.7,
                 use_label_smooth = False
                 ):
        self.losses = []
        dowmsample = 32

        for i, out in enumerate(outputs):
            """
            paddle.vision.ops.yolo_loss(x, gt_box, gt_label, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, gt_score=None, use_label_smooth=True, name=None, scale_x_y=1.0)

            x: 输出特征图。
            gt_box: 真实框。
            gt_label: 真实框标签。
            ignore_thresh，预测框与真实框IoU阈值超过ignore_thresh时，不作为负样本，YOLOv3模型里设置为0.7。
            downsample_ratio，特征图P0的下采样比例，使用Darknet53骨干网络时为32。
            gt_score，真实框的置信度，在使用了mixup技巧时用到。
            use_label_smooth，一种训练技巧，如不使用，设置为False。
            name，该层的名字，比如'yolov3_loss'，默认值为None，一般无需设置
            """
            anchors_masks_i = anchors_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                x=out,
                gt_box=gtbox,
                gt_label=gtlabel,
                gt_score=gtscore,
                anchors=anchors,
                anchor_mask=anchors_masks_i,
                class_num=self.num_classes,
                ignore_thresh=ignore_thresh,
                downsample_ratio=dowmsample,
                use_label_smooth=False
            )
            self.losses.append(paddle.mean(loss))
            dowmsample = dowmsample // 2

        return sum(self.losses)


if __name__ == '__main__':

    train_path = './insects/insects/train'
    valid_path = './insects/insects/val'
    eval_path = './insects/insects/test'

    train_dataset = TrainDataset(train_path, mode='train')
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    img, gt_boxes, gt_labels, img_shape = next(train_loader())

    print("img_shape", img.shape)
    print("box_shape", gt_boxes.shape)
    print("gt_labels", gt_labels.shape)
