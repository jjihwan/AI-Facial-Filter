import facer
import sys
import torch
import numpy as np
sys.path.append('..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

face_parser = facer.face_parser('farl/lapa/448', device=device)
face_detector = facer.face_detector('retinaface/mobilenet', device=device)


def FaRL_256(image):
    np_image = np.array(image.resize((256, 256)).convert('RGB'))
    image = facer.hwc2bchw(torch.from_numpy(np_image))
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    # seg_probs = torch.round(seg_logits.softmax(dim=1),decimals=0)
    seg_probs = torch.argmax(seg_logits, dim=1)[0]
    output = torch.zeros(1, 5, image.shape[2], image.shape[3])
    output[0, 0] = torch.where(seg_probs == 1, 1, 0)
    output[0, 2] = torch.where(seg_probs == 2, 1, 0) + \
        torch.where(seg_probs == 4, 1, 0)
    output[0, 1] = torch.where(seg_probs == 3, 1, 0) + \
        torch.where(seg_probs == 5, 1, 0)
    output[0, 4] = torch.where(seg_probs == 7, 1, 0) + torch.where(
        seg_probs == 8, 1, 0) + torch.where(seg_probs == 9, 1, 0)
    output[0, 3] = torch.where(seg_probs == 6, 1, 0)
    return output


def FaRL_1024(image):
    np_image = np.array(image.convert('RGB'))
    image = facer.hwc2bchw(torch.from_numpy(np_image))
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    # seg_probs = torch.round(seg_logits.softmax(dim=1),decimals=0)
    seg_probs = torch.argmax(seg_logits, dim=1)[0]
    output = torch.zeros(1, 5, image.shape[2], image.shape[3])
    output[0, 0] = torch.where(seg_probs == 1, 1, 0)
    output[0, 2] = torch.where(seg_probs == 2, 1, 0) + \
        torch.where(seg_probs == 4, 1, 0)
    output[0, 1] = torch.where(seg_probs == 3, 1, 0) + \
        torch.where(seg_probs == 5, 1, 0)
    output[0, 4] = torch.where(seg_probs == 7, 1, 0) + torch.where(
        seg_probs == 8, 1, 0) + torch.where(seg_probs == 9, 1, 0)
    output[0, 3] = torch.where(seg_probs == 6, 1, 0)
    return output

def FaRL_512(image):
    np_image = np.array(image.resize((512, 512)).convert('RGB'))
    image = facer.hwc2bchw(torch.from_numpy(np_image))
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_parser(image, faces)
    seg_logits = faces['seg']['logits']
    # seg_probs = torch.round(seg_logits.softmax(dim=1),decimals=0)
    seg_probs = torch.argmax(seg_logits, dim=1)[0]
    output = torch.zeros(1, 5, image.shape[2], image.shape[3])
    output[0, 0] = torch.where(seg_probs == 1, 1, 0)
    output[0, 2] = torch.where(seg_probs == 2, 1, 0) + \
        torch.where(seg_probs == 4, 1, 0)
    output[0, 1] = torch.where(seg_probs == 3, 1, 0) + \
        torch.where(seg_probs == 5, 1, 0)
    output[0, 4] = torch.where(seg_probs == 7, 1, 0) + torch.where(
        seg_probs == 8, 1, 0) + torch.where(seg_probs == 9, 1, 0)
    output[0, 3] = torch.where(seg_probs == 6, 1, 0)
    return output
