import cv2
import torch
import numpy as np


def zmMinFilterGray(src, r=7):
    '''Minimum filtering, r is the filter radius'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def calculate_transmission(m, r=81, eps=0.001, w=0.95, maxV1=0.80):  # Enter RGB image, value range [0,1]
    '''Calculating Atmospheric Mask Image V1 (Transmission Map)'''
     # Dark channel images are obtained
    V1,indexs = m.min(dim=1)
    V1 = V1.cpu().numpy()
    Dark_Channel = zmMinFilterGray(V1, 7)
    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # Optimized with bootstrap filtering
    V1 = np.minimum(V1 * w, maxV1)  
    V1 = np.expand_dims(V1, axis=1)
    return torch.from_numpy(V1)
     
