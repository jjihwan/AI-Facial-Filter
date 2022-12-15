import math
import torch
import numpy as np
from PIL import Image


def compute_h(p1, p2):
    # TODO ...
    # initialize
    N = p1.shape[0]
    one = np.ones((N, 1))
    p2 = np.concatenate((p2, one), axis=1)
    A = np.zeros((2*N, 6))
    b = np.reshape(p1, (2*N, 1))

    # set A, ATA
    A[range(0, 2*N, 2), :3] = p2
    A[range(1, 2*N, 2), 3:6] = p2

    ATA = A.T @ A
    ATb = A.T @ b

    t = np.linalg.pinv(ATA) @ ATb

    # find t corresponds to smallest eigenvalue
    H = np.concatenate((t.reshape((2, 3)), np.array([[0, 0, 1]])), axis=0)
    return H


def compute_h_norm(p1, p2):
    # TODO ...
    # initialize
    N = p1.shape[0]
    one = np.ones((N, 1))
    p1 = np.concatenate((p1, one), axis=1)
    p2 = np.concatenate((p2, one), axis=1)
    A = np.zeros((2*N, 9))

    # Normalize
    m1 = np.mean(p1, axis=0)
    m2 = np.mean(p2, axis=0)
    s1 = np.sqrt(2)/np.mean(np.sqrt(np.sum((p1-m1)*(p1-m1), axis=1)))
    s2 = np.sqrt(2)/np.mean(np.sqrt(np.sum((p2-m2)*(p2-m2), axis=1)))
    N1 = s1 * np.array([[1, 0, -m1[0]], [0, 1, -m1[1]], [0, 0, 1/s1]])
    N2 = s2 * np.array([[1, 0, -m2[0]], [0, 1, -m2[1]], [0, 0, 1/s2]])
    p1 = p1@N1.T
    p2 = p2@N2.T

    p1 = p1[:, :2]
    p2 = p2[:, :2]

    H_til = compute_h(p1, p2)
    H = np.linalg.pinv(N1)@H_til@N2

    return H


def warp_image(origin, H):
    # TODO ...
    print(origin.shape)
    _, X, Y = origin.shape
    affined = np.zeros((3, X, Y))

    H_inv = np.linalg.pinv(H)
    for i in range(X):
        for j in range(Y):
            p = np.array([[i], [j], [1]])
            q = H_inv@p
            x = round((q[0]/q[2]).item())
            y = round((q[1]/q[2]).item())
            if 0 <= x < X and 0 <= y < Y:
                affined[:, i, j] = origin[:, x, y]
    return affined


def warp_mask(origin, H):
    # TODO ...
    print(origin.shape)
    _, _, X, Y = origin.shape
    affined = np.zeros((1, 5, X, Y))

    H_inv = np.linalg.pinv(H)
    for i in range(X):
        for j in range(Y):
            p = np.array([[i], [j], [1]])
            q = H_inv@p
            x = round((q[0]/q[2]).item())
            y = round((q[1]/q[2]).item())
            if 0 <= x < X and 0 <= y < Y:
                affined[:, :, i, j] = origin[:, :, x, y]
    return affined


def set_cor(mask):
    """_summary_

    Args:
        mask (Tensor): mask 1024*1024

    Returns:
        _type_: _description_
    """
    mask = mask.numpy()
    n = mask.shape[0]
    l_eye = mask[0, 2]
    r_eye = mask[0, 1]
    lips = mask[0, 4]
    nose = mask[0, 3]

    l_eye_xcor = np.arange(1024).reshape((-1, 1)) * l_eye
    l_eye_ycor = np.arange(1024).reshape((1, -1)) * l_eye
    r_eye_xcor = np.arange(1024).reshape((-1, 1)) * r_eye
    r_eye_ycor = np.arange(1024).reshape((1, -1)) * r_eye
    lips_xcor = np.arange(1024).reshape((-1, 1)) * lips
    lips_ycor = np.arange(1024).reshape((1, -1)) * lips
    nose_xcor = np.arange(1024).reshape((-1, 1)) * nose
    nose_ycor = np.arange(1024).reshape((1, -1)) * nose

    lex = np.mean(l_eye_xcor[l_eye_xcor > 0])
    ley = np.mean(l_eye_ycor[l_eye_ycor > 0])
    rex = np.mean(r_eye_xcor[r_eye_xcor > 0])
    rey = np.mean(r_eye_ycor[r_eye_ycor > 0])
    lx = np.mean(lips_xcor[lips_xcor > 0])
    ly = np.mean(lips_ycor[lips_ycor > 0])
    nx = np.mean(nose_xcor[nose_xcor > 0])
    ny = np.mean(nose_ycor[nose_ycor > 0])

    origin = np.array([[lex, ley], [rex, rey], [lx, ly], [nx, ny]])
    print(origin)
    # left_eye, right_eye, lips, nose
    affined = 4*np.array([[110, 90], [110, 165], [197, 128], [143, 128]])
    return origin, affined


def affine_img_1024(img, mask):
    """_summary_

    Args:
        img (Tensor): 1024*1024
        mask (Tensor): 1024*1024

    Returns:
        affined_image(numpy): 1024*1024
    """

    origin_img = img  # .numpy()
    original_cor, affined_cor = set_cor(mask=mask)

    H = compute_h_norm(p1=affined_cor, p2=original_cor)

    affined_img = warp_image(origin=origin_img, H=H)

    return affined_img  # torch.from_numpy(affined_img)


def find_invH(img, mask):
    origin_img = img.numpy()
    original_cor, affined_cor = set_cor(mask=mask)

    H = compute_h_norm(p1=affined_cor, p2=original_cor)
    H_inv = np.linalg.pinv(H)
    return H_inv


def affine_mask(mask):
    origin_mask = mask.numpy()
    original_cor, affined_cor = set_cor(mask=mask)
    H = compute_h_norm(p1=affined_cor, p2=original_cor)

    affined_mask = warp_mask(origin=origin_mask, H=H)

    return torch.from_numpy(affined_mask)


if __name__ == '__main__':
    fileName = 'iu.jpg'
    mat_pil_img = Image.open(fileName)
    mat_img = mat_pil_img.resize((1024, 1024))
    affined_img = affine_img_1024(mat_img)
    Image.write('./affined_img.jpg')
