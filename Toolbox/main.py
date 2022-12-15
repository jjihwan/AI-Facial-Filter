import os
import sys
import cv2
import time
import numpy as np
import PIL
from PIL import Image
from PIL.ImageQt import ImageQt
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
from core.affine_1024 import affine_img_1024
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
        self.alpha = 0.7
        self.ref = None
        self.star = None  # selecting style. 0 - IU, 1 - JWY, 2 - NY
        self.star_style = torch.cat((torch.cat((torch.tensor([[-6.2113e-01,  2.0517e+00,  2.4000e-01, -2.6255e-01,  2.4179e-01,
                                               -2.8793e+00,  1.5588e+00, -9.0695e-01, -1.2933e+00,  2.9646e+00,
                                               5.5971e-01, -1.2251e+00, -9.0674e-01,  1.8832e+00,  1.6605e+00,
                                               -8.7556e-01,  2.5428e+00,  9.0850e-01,  1.2170e-01,  1.6328e-01,
                                               3.7637e+00, -5.3771e-01, -6.2790e-01, -2.0342e+00,  1.3229e+00,
                                               -1.0477e+00, -4.7527e-02, -3.4922e+00, -4.0181e-01,  1.3703e+00,
                                               8.1669e-01,  1.8076e+00,  5.7517e-01, -1.4479e-01, -1.8549e+00,
                                               -1.2505e+00, -1.8515e-02,  4.7660e-01, -4.1602e-01,  7.8310e-01,
                                               1.1340e+00, -1.4290e+00, -1.5522e+00,  2.3367e+00,  3.5454e-03,
                                               -9.5534e-01,  1.6768e+00,  2.0681e+00,  1.2742e+00, -3.8776e-01,
                                               3.7261e-01, -3.5170e-01,  7.0199e-01,  3.0009e+00,  2.4033e-02,
                                               -9.8368e-01, -1.6568e+00, -1.7119e+00, -8.1332e-01,  1.0358e-01,
                                               1.4604e+00,  2.3371e-01,  1.2524e+00, -2.9008e+00]]),
                                     torch.tensor([[-0.6654,  1.9051,  0.5576, -0.1215,  0.5871, -2.9469,  1.4394, -1.1295,
                                                   -1.5324,  3.3035,  0.2253, -1.1955, -1.0215,  2.0417,  1.5647, -0.8674,
                                                   2.4636,  0.5203,  0.2079,  0.1077,  3.8872, -0.7178, -0.3891, -2.3430,
                                                   1.7837, -0.9503, -0.3683, -3.8540, -0.3708,  1.6032,  0.9604,  2.0024,
                                                   0.5748, -0.0219, -2.1755, -1.2537,  0.1267,  0.4426, -0.6324,  0.8908,
                                                   1.0365, -1.9384, -1.3756,  2.7352,  0.2581, -1.2037,  1.8737,  1.8325,
                                                   1.2466, -0.2322,  0.5378, -0.1024,  0.4774,  2.9384, -0.1355, -0.5301,
                                                   -1.5025, -1.5791, -0.5855, -0.0762,  1.5727, -0.0438,  1.5195, -2.8828]])
                                                ), axis=0),
                                     torch.tensor([[-0.7206,  1.8766,  0.7194, -0.3504,  0.3427, -2.1000,  2.2526, -1.4914,
                                                    -1.6249,  2.3412,  0.8200, -1.0424, -0.8252,  1.7793,  2.6782,  0.0878,
                                                    2.1406,  1.0943,  0.1039,  0.6568,  3.1117,  0.1684, -1.0511, -0.9694,
                                                    1.3676, -0.2686,  0.0504, -2.3933, -0.3545,  1.2875,  0.6429,  1.8204,
                                                    -0.1296,  0.1761, -1.5760, -0.8816, -0.1128,  0.2682,  0.0153,  0.9614,
                                                    1.6165, -0.8339, -1.9320,  1.3848, -1.1752, -1.2134,  0.9706,  1.5553,
                                                    1.0308,  0.2278, -0.2469, -0.5066,  0.7989,  2.6576,  0.4700, -0.2395,
                                                    -1.4192, -2.1337, -0.2350,  0.4855,  1.1469,  0.4624,  1.2890, -1.6510]])), axis=0)

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

        self.affimg_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.affimg_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.affmask_scene = GraphicsScene(self.mode, self.size)
        self.graphicsView_3.setScene(self.affmask_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.ref_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # gout_scene / self.mode, self.size delete needed / Graph~ to QGraph
        self.gout_scene = QGraphicsScene()
        self.graphicsView_5.setScene(self.gout_scene)
        self.graphicsView_5.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_5.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_5.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.blended_scene = QGraphicsScene()
        self.graphicsView_6.setScene(self.blended_scene)
        self.graphicsView_6.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_6.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_6.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # final_scene
        self.result_scene = QGraphicsScene()
        self.graphicsView_7.setScene(self.result_scene)
        self.graphicsView_7.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_7.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_7.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.color = None

        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)

        self.slidern_1.setEnabled(False)

        pixmap1 = QPixmap('./iu.jpg')
        pixmap1 = pixmap1.scaled(120, 120, Qt.IgnoreAspectRatio)
        icon1 = QIcon()
        icon1.addPixmap(pixmap1)
        self.pushButton_6.setIcon(icon1)
        self.pushButton_6.setIconSize(QSize(120, 120))

        pixmap2 = QPixmap('./jang.jpg')
        pixmap2 = pixmap2.scaled(120, 120, Qt.IgnoreAspectRatio)
        icon2 = QIcon()
        icon2.addPixmap(pixmap2)
        self.pushButton_7.setIcon(icon2)
        self.pushButton_7.setIconSize(QSize(120, 120))

        pixmap3 = QPixmap('./nayeon.jpg')
        pixmap3 = pixmap3.scaled(120, 120, Qt.IgnoreAspectRatio)
        icon3 = QIcon()
        icon3.addPixmap(pixmap3)
        self.pushButton_8.setIcon(icon3)
        self.pushButton_8.setIconSize(QSize(120, 120))

    def slider(self, value):
        self.alpha = value/100.

    def open(self):
        self.load = True

        # celeb reference button activated
        self.pushButton_6.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.slidern_1.setEnabled(True)

        # loading real image
        directory = os.path.join(QDir.currentPath(), "samples/faces")
        directory_mask = os.path.join(QDir.currentPath(), "samples/masks")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", directory)

        # load skin mask
        tmp = fileName.split("/")[-1]
        tmp_filename = tmp.split(".")[0].zfill(5)

        if fileName:
            image = QPixmap(fileName)
            mat_pil_img = Image.open(fileName)
            mat_img = mat_pil_img.resize((1024, 1024))

            self.tensor_image_256 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                transforms.ToTensor()(mat_img.resize((256, 256))))
            self.tensor_image_1024 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                transforms.ToTensor()(mat_img))

            self.src_blend = mat_img

            mask_pred_1024 = FaRL_1024(mat_img)
            mask_pred_256 = F.interpolate(
                mask_pred_1024, (256, 256), mode='nearest')

            self.mask_pred_1024 = mask_pred_1024
            self.mask_pred_256 = mask_pred_256
            self.invH = find_invH(self.tensor_image_256, mask_pred_256)
            skin = torch.sum(mask_pred_1024[0, :], axis=0)

            self.mask_blend = transforms.ToPILImage()(skin)

            self.skin = skin

            skin = F.interpolate(torch.unsqueeze(torch.unsqueeze(
                skin, 0), 0), size=mat_img.size, mode='bilinear')
            skin = torch.squeeze(skin)
            skin = torch.unsqueeze(skin, 2)
            skin = skin.numpy()
            ###############
            mat_img = mat_img*skin.astype(np.uint8)
            mat_img = Image.fromarray(mat_img)

            self.img = mat_img.copy()

            affine_s = affine_img(transforms.ToTensor()(
                mat_img.resize((256, 256))), mask_pred_256).unsqueeze(0)

            result = affine_s.permute(0, 2, 3, 1)
            result = result.detach().cpu().numpy()
            result = result.clip(0, 1)
            result = result * 255

            result = np.asarray(result[0, :, :, :], dtype=np.uint8)
            result = result.copy()
            qim = QImage(result.data, 256, 256, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName)
                return
            image = image.scaled(
                self.graphicsView_1.size(), Qt.IgnoreAspectRatio)

            if len(self.scene.items()) > 0:
                self.scene.removeItem(self.scene.items()[-1])
            self.scene.addPixmap(image)\

            if len(self.affimg_scene.items()) > 0:
                self.affimg_scene.removeItem(self.affimg_scene.items()[-1])
            self.affimg_scene.addPixmap(QPixmap.fromImage(qim))

        self.mask_name = fileName
        self.load_mask()

    def load_mask(self):
        # loading mask from real image
        skin_mask = np.zeros([256, 256])
        nose_mask = np.zeros([256, 256])
        eye_mask = np.zeros([256, 256])
        mouth_mask = np.zeros([256, 256])

        # affine mask
        mask_pred = affine_mask(self.mask_pred_256)

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

        # resize mask
        eye_mask = eye_mask.clone().detach()
        size_tmp = int(self.mask_size[0] * eye_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(
            eye_mask, 0), 0), size=size_tmp, mode='bilinear')
        eye_mask = kornia.geometry.transform.center_crop(mask_tmp, (256, 256))[
            0, 0]
        eye_mask = eye_mask.numpy()
        eye_mask[eye_mask != 0] = 1

        mouth_mask = mouth_mask.clone().detach()
        size_tmp = int(self.mask_size[1] * mouth_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(
            mouth_mask, 0), 0), size=size_tmp, mode='bilinear')
        mouth_mask = kornia.geometry.transform.center_crop(mask_tmp, (256, 256))[
            0, 0]
        mouth_mask = mouth_mask.numpy()
        mouth_mask[mouth_mask != 0] = 2

        nose_mask = nose_mask.clone().detach()
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
            self.graphicsView_3.size(), Qt.IgnoreAspectRatio)
        self.affmask_scene.reset()
        if len(self.affmask_scene.items()) > 0:
            self.affmask_scene.reset_items()
        self.affmask_scene.addPixmap(self.image)

    def open_ref(self):
        self.star = None
        if self.load == True:
            self.pushButton_4.setEnabled(True)
            self.pushButton_5.setEnabled(True)
            self.pushButton_7.setEnabled(True)
            self.pushButton_8.setEnabled(True)

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
                self.mask_m, self.affmask_scene.mask_points[i], self.affmask_scene.size_points[i], i)

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

        mask = transform_mask(Image.fromarray(np.uint8(mask)))
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = affine_img(transform_image(img), self.mask_pred_256)

        start_t = time.time()

        # mix the styles
        if self.star is None:
            ref = transform_image(self.ref.copy())
            ref_style = style_encoder(torch.FloatTensor(
                [ref.numpy()]), torch.LongTensor([self.y_trg]))
        else:
            ref_style = self.star_style[self.star, :]
        s_style = style_encoder(torch.FloatTensor(
            [img.numpy()]), torch.LongTensor([self.y_trg]))
        s_trg = self.alpha * ref_style + (1-self.alpha) * s_style

        # generating
        masks = (torch.FloatTensor([mask_m.numpy()]),
                 torch.FloatTensor([mask.numpy()]))
        generated = generator(torch.FloatTensor(
            [img.numpy()]), s_trg, masks=masks, attribute=self.attribute)
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))

        # blending
        generated_blend = torch.from_numpy(
            warp_image(generated.detach().numpy().squeeze(), self.invH))

        generated_blend = F.interpolate(generated_blend.unsqueeze(0), (1024, 1024), mode='bilinear') * \
            torch.unsqueeze(torch.unsqueeze(self.skin, 0),
                            0) + self.tensor_image_1024 * torch.unsqueeze((1-self.skin), 0)

        # optimizing
        self.blended = transforms.ToPILImage()(
            torch.squeeze(((generated_blend+1)/2).clip(0, 1), 0))

        self.src_blend.save("src.jpg")
        self.mask_blend.save("mask.jpg")
        self.blended.save("blended.jpg")

        src_blend = img_as_float(imread("src.jpg"))
        blended = img_as_float(imread("blended.jpg"))
        mask_blend = imread("mask.jpg", as_gray=True).astype(blended.dtype)/255

        opt_im = blend_optimize(src_blend, blended, mask_blend, 128, color_weight=1,
                                gradient_kernel='normal', n_iteration=1, whole_grad=True, origin_res=True)

        # generated output
        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()
        result = (result + 1) / 2
        result = result.clip(0, 1)
        result = result * 255

        result = np.asarray(result[0, :, :, :], dtype=np.uint8)
        result = result.copy()

        # blended output
        result_blend = generated_blend.permute(0, 2, 3, 1)
        result_blend = result_blend.detach().cpu().numpy()
        result_blend = (result_blend + 1) / 2
        result_blend = result_blend.clip(0, 1)
        result_blend = result_blend * 255

        result_blend = np.asarray(result_blend[0, :, :, :], dtype=np.uint8)
        result_blend = result_blend.copy()

        # optimized output
        self.output_img = opt_im

        qim = QImage(opt_im, 1024, 1024, QImage.Format_RGB888).scaled(384, 384)
        qim1 = QImage(result.data, 256, 256, QImage.Format_RGB888)
        qim2 = QImage(result_blend.data, 1024, 1024,
                      QImage.Format_RGB888).scaled(256, 256)

        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))
        if len(self.gout_scene.items()) > 0:
            self.gout_scene.removeItem(self.gout_scene.items()[-1])
        self.gout_scene.addPixmap(QPixmap.fromImage(qim1))
        if len(self.blended_scene.items()) > 0:
            self.blended_scene.removeItem(self.blended_scene.items()[-1])
        self.blended_scene.addPixmap(QPixmap.fromImage(qim2))

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

    def celeb1(self):
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)

        self.star = 0

        celebimg1 = QPixmap('./iu.jpg')
        celebimg1 = celebimg1.scaled(384, 384, Qt.IgnoreAspectRatio)
        self.ref_scene.addPixmap(celebimg1)
        print("IU")
        return 0

    def celeb2(self):
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)

        self.star = 1

        celebimg2 = QPixmap('./jang.jpg')
        celebimg2 = celebimg2.scaled(384, 384, Qt.IgnoreAspectRatio)
        self.ref_scene.addPixmap(celebimg2)
        print("Jang Won Young")
        return 0

    def celeb3(self):
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)

        self.star = 2

        celebimg3 = QPixmap('./nayeon.jpg')
        celebimg3 = celebimg3.scaled(384, 384, Qt.IgnoreAspectRatio)
        self.ref_scene.addPixmap(celebimg3)
        print("Jeny")
        return 0

    def clear(self):
        self.pushButton_6.setEnabled(False)
        self.slidern_1.setEnabled(True)
        self.slidern_1.setValue(49)
        self.attribute = []
        self.mask_size = [1.0, 1.0, 1.0]
        self.tmp_attribute = None
        self.load_mask()


def _load_checkpoint(nets_ema, checkpoint_dir, step):
    ckptios = [CheckpointIO(
        ospj(checkpoint_dir, 'facial_checkpoint.ckpt'), **nets_ema)]
    for ckptio in ckptios:
        ckptio.load(step)


def blend_optimize(src, blended, mask, image_size=256, color_weight=1, gradient_kernel='normal',
                   n_iteration=2, whole_grad=False, origin_res=False):

    h_orig, w_orig, _ = src.shape
    blended = resize(blended, (h_orig, w_orig))
    mask = resize(mask, (h_orig, w_orig))

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
