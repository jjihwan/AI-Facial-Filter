import os
import sys
import cv2
import time
import numpy as np
import PIL
from PIL import Image
import math

import torch
from torchvision import transforms
from torchvision.utils import save_image

# OUR
import copy
import kornia
from munch import Munch

from os.path import join as ospj

import torch
import torch.nn.functional as F

from core.model import Generator, MappingNetwork, StyleEncoder
from core.unet import UNet
from core.FaRL_faceparser import FaRL_256, FaRL_1024
from core.checkpoint import CheckpointIO
from core.affine import affine_img, affine_mask, warp_image, find_invH

from ui import ui
from ui.mouse_event import GraphicsScene


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from skimage import img_as_float
from skimage.io import imread, imsave
from scipy.fftpack import dct, idct
from scipy.ndimage import correlate
from skimage.filters import gaussian, sobel_h, sobel_v, scharr_h, scharr_v, roberts_pos_diag, roberts_neg_diag, prewitt_h, prewitt_v
from skimage.transform import resize

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(
    76, 153, 0), QColor(204, 204, 0),  QColor(51, 51, 255)]


def normal_h(im): return correlate(
    im, np.asarray([[0, -1, 1]]), mode='nearest')


def normal_v(im): return correlate(
    im, np.asarray([[0, -1, 1]]).T, mode='nearest')


gradient_operator = {
    'normal': (normal_h, normal_v),  # default
    'sobel': (sobel_h, sobel_v),
    'scharr': (scharr_h, scharr_v),
    'roberts': (roberts_pos_diag, roberts_neg_diag),
    'prewitt': (prewitt_h, prewitt_v)
}


class Ex(QWidget, ui.Ui_Form):
    def __init__(self):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()

        self.output_img = None
        self.mat_img = None
        self.attribute = []
        self.tmp_attribute = None
        self.mask_size = [1.0, 1.0, 1.0]
        self.mask_name = ""
        self.load = False
        self.y_trg = 0
        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None
        self.skin = None
        self.tensor_image_256 = None
        self.tensor_image_1024 = None
        self.invH = None
        self.mask_pred_256 = None
        self.mask_pred_1024 = None
        self.alpha = 0.3

        self.src = None
        self.src_blend = None
        self.mask_blend = None
        self.blended = None

        self.mouse_clicked = False
        self.scene = QGraphicsScene()
        self.graphicsView_1.setScene(self.scene)
        self.graphicsView_1.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.mask_scene = GraphicsScene(self.mode, self.size)
        self.graphicsView_2.setScene(self.mask_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.ref_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.result_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.color = None

        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.pushButton_11.setEnabled(False)
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(False)

        #self.mask_scene.size = 6

    def slider(self, value):
        # mapping
        mask_size = 0.85 + 0.003*value
        self.mask_size[self.tmp_attribute] = mask_size
        self.load_mask()

    def open(self):
        self.load = True
        # loading real image
        directory = os.path.join(QDir.currentPath(), "samples/faces")
        directory_mask = os.path.join(QDir.currentPath(), "samples/masks")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", directory)

        # load skin mask
        tmp = fileName.split("/")[-1]
        tmp_filename = tmp.split(".")[0].zfill(5)

        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            mat_img = mat_img.resize((1024, 1024))

            self.tensor_image_256 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                transforms.ToTensor()(mat_img.resize((256, 256))))
            self.tensor_image_1024 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                transforms.ToTensor()(mat_img))
            self.src_blend = transforms.ToPILImage()((self.tensor_image_1024+1)/2)

            mask_pred_1024 = FaRL_1024(mat_img)
            mask_pred_256 = FaRL_256(mat_img)

            self.mask_pred_1024 = mask_pred_1024
            self.mask_pred_256 = mask_pred_256
            affined_img = affine_img(self.tensor_image_256, mask_pred_256)
            transforms.ToPILImage()(affined_img).show()
            self.invH = find_invH(self.tensor_image_256, mask_pred_256)
            skin = torch.sum(mask_pred_1024[0, :], axis=0)

            self.mask_blend = transforms.ToPILImage()(skin)

            self.skin = skin

            skin = F.interpolate(torch.unsqueeze(torch.unsqueeze(
                skin, 0), 0), size=mat_img.size, mode='bilinear')
            skin = torch.squeeze(skin)
            skin = torch.unsqueeze(skin, 2)
            skin = skin.numpy()

            mat_img = mat_img*skin.astype(np.uint8)
            mat_img = Image.fromarray(mat_img)

            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName)
                return
            image = image.scaled(
                self.graphicsView_1.size(), Qt.IgnoreAspectRatio)

            if len(self.scene.items()) > 0:
                self.scene.removeItem(self.scene.items()[-1])
            self.scene.addPixmap(image)

        self.mask_name = fileName
        self.load_mask(mask_pred_256)

    def load_mask(self, mask_pred):
        # loading mask from real image
        skin_mask = np.zeros([256, 256])
        nose_mask = np.zeros([256, 256])
        eye_mask = np.zeros([256, 256])
        mouth_mask = np.zeros([256, 256])

        mask_pred = affine_mask(mask_pred)

        s = mask_pred[0, 0]
        e = mask_pred[0, 1] + mask_pred[0, 2]
        m = mask_pred[0, 3]
        n = mask_pred[0, 4]

        eye_mask = e
        mouth_mask = 2*m
        nose_mask = 3*n
        skin_mask = 4*s

        tmp = self.mask_name.split("/")[-1]
        tmp_filename = tmp.split(".")[0].zfill(5)

        directory = os.path.join(QDir.currentPath(), "samples/masks")

        # resikze mask
        eye_mask = torch.tensor(eye_mask)
        size_tmp = int(self.mask_size[0] * eye_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(
            eye_mask, 0), 0), size=size_tmp, mode='bilinear')
        eye_mask = kornia.geometry.transform.center_crop(mask_tmp, (256, 256))[
            0, 0]
        eye_mask = eye_mask.numpy()
        eye_mask[eye_mask != 0] = 1

        mouth_mask = torch.tensor(mouth_mask)
        size_tmp = int(self.mask_size[1] * mouth_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(
            mouth_mask, 0), 0), size=size_tmp, mode='bilinear')
        mouth_mask = kornia.geometry.transform.center_crop(mask_tmp, (256, 256))[
            0, 0]
        mouth_mask = mouth_mask.numpy()
        mouth_mask[mouth_mask != 0] = 2

        nose_mask = torch.tensor(nose_mask)
        size_tmp = int(self.mask_size[2] * nose_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(
            nose_mask, 0), 0), size=size_tmp, mode='bilinear')
        nose_mask = kornia.geometry.transform.center_crop(mask_tmp, (256, 256))[
            0, 0]
        nose_mask = nose_mask.numpy()
        nose_mask[nose_mask != 0] = 3

        tmp_mask = skin_mask + eye_mask
        tmp_mask[tmp_mask > 4] = 1
        tmp_mask = tmp_mask + mouth_mask
        tmp_mask[tmp_mask > 5] = 2
        tmp_mask = tmp_mask + nose_mask
        tmp_mask[tmp_mask > 4] = 3

        res_mask = np.repeat(tmp_mask[:, :, np.newaxis], 3, axis=2)
        res_mask = np.asarray(res_mask, dtype=np.uint8)

        self.mask = res_mask.copy()
        self.mask_m = res_mask
        mat_img = res_mask.copy()

        image = QImage(mat_img.data, 256, 256, QImage.Format_RGB888)

        for i in range(256):
            for j in range(256):
                r, g, b, a = image.pixelColor(i, j).getRgb()
                image.setPixel(i, j, color_list[r].rgb())

        pixmap = QPixmap()
        pixmap.convertFromImage(image)
        self.image = pixmap.scaled(
            self.graphicsView_1.size(), Qt.IgnoreAspectRatio)
        self.mask_scene.reset()
        if len(self.mask_scene.items()) > 0:
            self.mask_scene.reset_items()
        self.mask_scene.addPixmap(self.image)

    def open_ref(self):
        if self.load == True:
            self.pushButton_4.setEnabled(True)
            self.pushButton_5.setEnabled(True)
            self.pushButton_7.setEnabled(True)
            self.pushButton_8.setEnabled(True)
            self.pushButton_9.setEnabled(True)
            self.pushButton_10.setEnabled(True)
            self.pushButton_11.setEnabled(True)

        # loading random reference image for style
        directory = os.path.join(QDir.currentPath(), "samples/faces")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", directory)
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.ref = mat_img.copy()
            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName)
                return

            image = image.scaled(
                self.graphicsView_1.size(), Qt.IgnoreAspectRatio)
            if len(self.ref_scene.items()) > 0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)

    def apply(self):
        self.pushButton_3.setEnabled(True)

        # updates the changes of the mask
        for i in range(5):
            self.mask_m = self.make_mask(
                self.mask_m, self.mask_scene.mask_points[i], self.mask_scene.size_points[i], i)

        transform_mask = transforms.Compose([transforms.Resize([256, 256]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0, 0, 0), (1 / 255., 1 / 255., 1 / 255.))
                                             ])

        transform_image = transforms.Compose([transforms.Resize([256, 256]),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])
        mask = self.mask.copy()
        mask_m = self.mask_m.copy()
        img = self.img.copy()
        ref = self.ref.copy()

        mask = transform_mask(Image.fromarray(np.uint8(mask)))
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = affine_img(transform_image(img), self.mask_pred_256)
        #img = transform_image(img)

        # transforms.ToPILImage()(img).show()
        ref = transform_image(ref)

        start_t = time.time()

        s_trg = self.alpha * style_encoder(torch.FloatTensor(
            [ref.numpy()]), torch.LongTensor([self.y_trg])) + (1-self.alpha) * style_encoder(torch.FloatTensor(
                [img.numpy()]), torch.LongTensor([self.y_trg]))

        # s_trg = style_encoder(torch.FloatTensor(
        #     [ref.numpy()]), torch.LongTensor([self.y_trg]))
        masks = (torch.FloatTensor([mask_m.numpy()]),
                 torch.FloatTensor([mask.numpy()]))
        generated = generator(torch.FloatTensor(
            [img.numpy()]), s_trg, masks=masks, attribute=self.attribute)
        # print(mask.shape)
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        generated = torch.from_numpy(
            warp_image(generated.detach().numpy().squeeze(), self.invH))

        generated = F.interpolate(generated.unsqueeze(0), (1024, 1024), mode='bilinear') * \
            torch.unsqueeze(torch.unsqueeze(self.skin, 0),
                            0) + self.tensor_image_1024 * torch.unsqueeze((1-self.skin), 0)

        self.blended = transforms.ToPILImage()(torch.squeeze((generated+1)/2, 0))

        self.src_blend.save("src.jpg")
        self.mask_blend.save("mask.jpg")
        self.blended.save("blended.jpg")

        src_blend = img_as_float(imread("src.jpg"))
        blended = img_as_float(imread("blended.jpg"))
        mask_blend = imread("mask.jpg", as_gray=True).astype(blended.dtype)/255

        opt_im = blend_optimize(src_blend, blended, mask_blend, 128, color_weight=1,
                                gradient_kernel='normal', n_iteration=1, whole_grad=True, origin_res=True)

        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()
        result = (result + 1) / 2
        result = result.clip(0, 1)
        result = result * 255

        result = np.asarray(result[0, :, :, :], dtype=np.uint8)
        result = result.copy()

        self.output_img = opt_im

        qim = QImage(opt_im, 1024, 1024, QImage.Format_RGB888)

        # self.output_img = result

        # qim = QImage(result.data, 1024, 1024, QImage.Format_RGB888)
        qim = qim.scaled(256, 256)
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def make_mask(self, mask, pts, sizes, color):
        if len(pts) > 0:
            for idx, pt in enumerate(pts):
                cv2.line(mask, pt['prev'], pt['curr'],
                         (color, color, color), sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(
                self, "Save File", QDir.currentPath())
            try:
                im = Image.fromarray(self.output_img)
                im.save(fileName+'.jpg')
            except:
                pass

    def clear(self):
        self.pushButton_6.setEnabled(False)
        self.slidern_1.setEnabled(False)
        self.slidern_1.setValue(49)
        self.slidern_2.setEnabled(False)
        self.slidern_2.setValue(49)
        self.slidern_3.setEnabled(False)
        self.slidern_3.setValue(49)
        self.attribute = []
        self.mask_size = [1.0, 1.0, 1.0]
        self.tmp_attribute = None
        self.load_mask()

    def man_mode(self):
        self.y_trg = 1

    def woman_mode(self):
        self.y_trg = 0

    def eyes_mode(self):
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(True)
        self.slidern_3.setEnabled(False)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 1
        self.tmp_attribute = 0
        if (0 in self.attribute) == False:
            self.attribute.append(0)

    def mouth_mode(self):
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 2
        self.tmp_attribute = 1
        if (1 in self.attribute) == False:
            self.attribute.append(1)

    def nose_mode(self):
        self.slidern_1.setEnabled(True)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(False)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 3
        self.tmp_attribute = 2
        if (2 in self.attribute) == False:
            self.attribute.append(2)

    def skin_mode(self):
        self.mask_scene.mode = 4


def _load_checkpoint(nets_ema, checkpoint_dir, step):
    ckptios = [CheckpointIO(
        ospj(checkpoint_dir, 'facial_checkpoint.ckpt'), **nets_ema)]
    for ckptio in ckptios:
        ckptio.load(step)


def blend_optimize(src, blended, mask, image_size=256, color_weight=1, gradient_kernel='normal',
                   n_iteration=2, whole_grad=False, origin_res=False):

    h_orig, w_orig, _ = src.shape
    print("Blended Shape", blended.shape)
    blended = resize(blended, (h_orig, w_orig))
    mask = resize(mask, (h_orig, w_orig))
    print("Source Shape", src.shape)
    print("Blended Shape", blended.shape)
    print("Mask Shape", mask.shape)

    ############################ Image Gaussian Poisson Editing #############################
    if n_iteration > 1:  # in case for iterative optimization
        origin_color_weight = color_weight
        color_weight = 0.5

    final_grad = whole_grad
    for iter in range(n_iteration):
        if (n_iteration > 1) and (iter >= n_iteration-2):
            color_weight = origin_color_weight
            whole_grad = final_grad

        # pyramid
        max_level = int(math.ceil(np.log2(max(w_orig, h_orig) / image_size)))
        blended_im_pyramid = laplacian_pyramid(blended, max_level, image_size)
        src_im_pyramid = laplacian_pyramid(src, max_level, image_size)

        # init image
        # mask_init = ndarray_resize(mask, (image_size, image_size), order=0)[:, :, np.newaxis]
        # blended_init = fg_im_pyramid[0] * mask_init + bg_im_pyramid[0] * (1 - mask_init)
        blended_init = blended_im_pyramid[0]

        opt_im = np.clip(blended_init, 0, 1).astype(blended.dtype)
        # Start pyramid
        for level in range(max_level + 1):
            size = blended_im_pyramid[level].shape[:2]
            mask_im = ndarray_resize(mask, size, order=0)[
                :, :, np.newaxis, np.newaxis]
            if level != 0:
                opt_im = ndarray_resize(opt_im, size)
            opt_im = run_editing(src_im_pyramid[level], blended_im_pyramid[level],
                                 mask_im, opt_im, color_weight, gradient_kernel, whole_grad)
        if n_iteration > 1:
            src = opt_im

    opt_im = np.clip(opt_im * 255, 0, 255).astype(np.uint8)

    return opt_im


def laplacian_pyramid(im, max_level, image_size):
    im_pyramid = [im]
    for i in range(max_level - 1, -1, -1):
        im_pyramid_last = ndarray_resize(
            im_pyramid[-1], (image_size * 2 ** i, image_size * 2 ** i))
        im_pyramid.append(im_pyramid_last)

    im_pyramid.reverse()
    return im_pyramid


def ndarray_resize(im, image_size, order=3, dtype=None):
    im = resize(im, image_size, preserve_range=True,
                order=order, mode='constant')

    if dtype:
        im = im.astype(dtype)
    return im


def run_editing(src, blended_im, mask_im, opt_im, color_weight, gradient_kernel='normal', whole_grad=False):
    # get geometry/gradient feature
    if whole_grad:
        # source background texture
        bg_feature = gradient_feature(src, opt_im, gradient_kernel)
    else:
        bg_feature = gradient_feature(
            blended_im, opt_im, gradient_kernel)  # new background texture
    fg_feature = gradient_feature(src, opt_im, gradient_kernel)  # foreground
    feature = bg_feature * (1 - mask_im) + fg_feature * \
        mask_im  # combined gradient feature

    # get parameters
    size, dtype = feature.shape[:2], feature.dtype
    param_l = laplacian_param(size, dtype)  # gradient
    param_g = gaussian_param(size, dtype)  # color

    # run editing
    opt_im = gp_editing(feature, param_l, param_g, color_weight=color_weight)
    opt_im = np.clip(opt_im, 0, 1)

    return opt_im


def gradient_feature(im, color_feature, gradient_kernel):
    result = np.zeros((*im.shape, 5))

    gradient_h, gradient_v = gradient_operator[gradient_kernel]

    result[:, :, :, 0] = color_feature
    result[:, :, :, 1] = imfilter2d(im, gradient_h)
    result[:, :, :, 2] = imfilter2d(im, gradient_v)
    result[:, :, :, 3] = np.roll(result[:, :, :, 1], 1, axis=1)
    result[:, :, :, 4] = np.roll(result[:, :, :, 2], 1, axis=0)

    return result.astype(im.dtype)

# get laplacian filter (for image gradiant)


def laplacian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = np.roll(K, -(kw // 2), axis=0)
    K = np.roll(K, -(kh // 2), axis=1)

    return fft2(K, size, dtype)

# get gaussian


def gaussian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    K[1, 1] = 1
    # K[:3, :3] = gaussian(K[:3, :3], sigma) # actually do not apply gaussian filtering

    K = np.roll(K, -1, axis=0)
    K = np.roll(K, -1, axis=1)

    return fft2(K, size, dtype)


def gp_editing(X, param_l, param_g, color_weight=1, eps=1e-12):
    Fh = (X[:, :, :, 1] + np.roll(X[:, :, :, 3], -1, axis=1)) / 2
    Fv = (X[:, :, :, 2] + np.roll(X[:, :, :, 4], -1, axis=0)) / 2
    L = np.roll(Fh, 1, axis=1) + np.roll(Fv, 1, axis=0) - Fh - Fv

    param = param_l + color_weight * param_g
    param[(param >= 0) & (param < eps)] = eps
    param[(param < 0) & (param > -eps)] = -eps

    Y = np.zeros(X.shape[:3])
    for i in range(3):
        Xdct = dct2(X[:, :, i, 0])
        Ydct = (dct2(L[:, :, i]) + color_weight * Xdct) / param
        Y[:, :, i] = idct2(Ydct)
    return Y


def imfilter2d(im, filter_func):
    gradients = np.zeros_like(im)
    for i in range(im.shape[2]):
        gradients[:, :, i] = filter_func(im[:, :, i])

    return gradients


def fft2(K, size, dtype):
    w, h = size
    param = np.fft.fft2(K)
    param = np.real(param[0:w, 0:h])

    return param.astype(dtype)


def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T


if __name__ == '__main__':

    # hyper-parametrs
    checkpoint_dir = "checkpoints"
    resume_iter = 200000

    # initilize networks
    generator = Generator()
    mapping_network = MappingNetwork()
    style_encoder = StyleEncoder()
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network, style_encoder=style_encoder)
    # load weights
    _load_checkpoint(nets_ema, checkpoint_dir, resume_iter)

    app = QApplication(sys.argv)
    ex = Ex()
    sys.exit(app.exec_())
