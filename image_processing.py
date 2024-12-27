"""
   Based on code from Joerg Dietrich <astro@joergdietrich.com>. Copyright 2015-2020 under the GNU GPL version 2, see COPYING for details.
"""

import logging
import numpy as np
from PIL import Image
import math

# Set logging level to WARNING or higher
logging.getLogger("PIL").setLevel(logging.WARNING)

# Generic scores for prognosis
SCORES = {"protan": (0, 100, 100), "deutan": (100, 0, 100), "tritan": (100, 100, 0)}  # RGB


def transform_colourspace(img, mat):
    """ Transform images to another colour space based on user defined function

    Arguments:
    ----------
    img : array of shape (M, N, 3)
    mat : array of shape (3, 3)
        conversion matrix to different color space

    Returns:
    --------
    out : array of shape (M, N, 3)
    """
    return img @ mat.T

def simulate(linear_rgb, prognosis="deutan", severity = 0.0):
    """"
    Simulate the effect of colour blindness on an image using the model proposed by (MacHado & Oliveira & Fernandes, 2009).
    Note this model does not work well for tritanopia.

    Arguments:
    ----------
    linear_rgb : array of shape (M, N, 3)
        image in linearRGB format, values must be [0, 1] bounded
    prognosis : {"deutan", "protan", "tritan"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia

    Returns:
    --------
    sim_rgb : array of shape (M, N, 3)
        simulated image in RGB format
    """
    prog_matrices= {
    "protan" : {
        0: np.array([ [1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.002880, -0.001563, 1.004443] ]),
        2: np.array([ [0.734766, 0.334872, -0.069637], [0.051840, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137] ]),
        3: np.array([ [0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032] ]),
        4: np.array([ [0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095] ]),
        5: np.array([ [0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301] ]),
        6: np.array([ [0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632] ]),
        7: np.array([ [0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.077790], [-0.007025, -0.028051, 1.035076] ]),
        8: np.array([ [0.259411, 0.923008, -0.182420], [0.110296, 0.804340, 0.085364], [-0.006276, -0.034346, 1.040622] ]),
        9: np.array([ [0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265] ]),
        10: np.array([ [0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998] ])
    },

    "deutan": {
        0: np.array([ [1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.011370], [-0.003453, 0.007233, 0.996220] ]),
        2: np.array([ [0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702] ]),
        3: np.array([ [0.675425, 0.433850, -0.109275], [0.125303, 0.847755, 0.026942], [-0.007950, 0.018572, 0.989378] ]),
        4: np.array([ [0.605511, 0.528560, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.986200] ]),
        5: np.array([ [0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.010410, 0.027275, 0.983136] ]),
        6: np.array([ [0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162] ]),
        7: np.array([ [0.457771, 0.731899, -0.189670], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261] ]),
        8: np.array([ [0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421] ]),
        9: np.array([ [0.392952, 0.823610, -0.216562], [0.263559, 0.690210, 0.046232], [-0.011910, 0.040281, 0.971630] ]),
        10: np.array([ [0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881] ])
    },

    "tritan": {
        0: np.array([ [1.000000, 0.000000, -0.000000],  [ 0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000] ]),
        1: np.array([ [0.926670, 0.092514, -0.019184],  [ 0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.936750] ]),
        2: np.array([ [0.895720, 0.133330, -0.029050],  [ 0.029997, 0.945400, 0.024603], [0.013027, 0.104707, 0.882266] ]),
        3: np.array([ [0.905871, 0.127791, -0.033662],  [ 0.026856, 0.941251, 0.031893], [0.013410, 0.148296, 0.838294] ]),
        4: np.array([ [0.948035, 0.089490, -0.037526],  [ 0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156] ]),
        5: np.array([ [1.017277, 0.027029, -0.044306],  [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913] ]),
        6: np.array([ [1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742] ]),
        7: np.array([ [1.193214, -0.109812, -0.083402], [-0.058496, 0.979410, 0.079086], [-0.002346, 0.403492, 0.598854] ]),
        8: np.array([ [1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102] ]),
        9: np.array([ [1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838] ]),
        10: np.array([ [1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900] ])
    }
}
    # Interpolate the MacHado matrices
    severity_lower = int(math.floor(severity*10.0))
    severity_higher = min(severity_lower +1, 10)
    m1 = prog_matrices[prognosis][severity_lower]
    m2 = prog_matrices[prognosis][severity_higher]

    alpha = (severity - severity_lower/10.0)
    m = alpha*m2 + (1.0-alpha)*m1

    sim_cvd_linear_rgb = transform_colourspace(linear_rgb, m)

    return desaturate_linearRGB(sim_cvd_linear_rgb)

def simulate_tritan(rgb):
    """
    The MacHando et al. technique does not perfrom well for Tritan so it is better to make use of Vieron method for tritan

      Arguments:
        ----------
        rgb : array of shape (M, N, 3)
            original image in RGB format, values must be [0, 1] bounded
        color_deficit : {"d", "p", "t"}, optional
            type of colorblindness, d for deuteronopia (default),
            p for protonapia,
            t for tritanopia

      Returns:
      --------
        sim_rgb : array of shape (M, N, 3)
            simulated image in RGB format
    """

    prog_matrices = {
        "deutan": np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]], dtype=np.float16),
        "protan": np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]], dtype=np.float16),
        "tritan": np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]], dtype=np.float16),
    }
    rgb2lms = np.array([[0.3904725 , 0.54990437, 0.00890159],
       [0.07092586, 0.96310739, 0.00135809],
       [0.02314268, 0.12801221, 0.93605194]], dtype=np.float16)
    # Precomputed inverse
    lms2rgb = np.array([[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
       [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
       [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]], dtype=np.float16)
    # first go from RBG to LMS space
    lms = transform_colourspace(rgb, rgb2lms)
    # Calculate image as seen by the color blind
    sim_lms = transform_colourspace(lms, prog_matrices["tritan"])
    # Transform back to RBG
    sim_cvd_linear_rgb = transform_colourspace(sim_lms, lms2rgb)
    return desaturate_linearRGB(sim_cvd_linear_rgb)

def desaturate_linearRGB(im):
    """
    Make RGB colors fit in the [0,1] range by desaturating after simulation.

    Inspired from https://github.com/DaltonLens/DaltonLens-Python/blob/master/daltonlens
    Instead of just clipping to 0,1, we move the color towards
    the white point, desaturating it until it fits the RGB gamut.

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input linear RGB image

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output linear RGB image with values in [0, 1]
    """
    # Find the most negative value for each, or 0
    min_val = np.fmin(im[:,0], im[:,1])
    min_val = np.fmin(min_val, im[:,2])
    min_val = np.fmin(min_val, 0.0)
    # Add the same white component to all 3 values until none
    # is negative and clip the max of each component to 1.0
    return np.clip(im - min_val[:,np.newaxis], 0., 1.0)

def daltonize(linear_rgb, prognosis="deutan", sev = 1.0):
    """
    Adjust colours of an imafe to compensate for CVD prognosis

     Arguments:
    ----------
    linear_rgb : array of shape (M, N, 3)
        original image in RGB format, values must be [0, 1] bounded
    prognosis : {"deutan", "protan", "tritan"}, optional
        type of colorblindness, deutan for deuteronopia (default),
        protan for protonapia,
        tritan for tritanopia

    Returns:
    --------
    dtpn : array of shape (M, N, 3)
        image in RGB format with colors adjusted
    """
    if prognosis == "tritan":
        sim_rgb = simulate_tritan(linear_rgb)
    else:
        sim_rgb = simulate(linear_rgb, prognosis, sev)

    err2mod = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    # Matrix aims to transform the lost information in way that shifts colours towards colours that dichromat can see
    # Emperically tuned
    err = transform_colourspace((linear_rgb - sim_rgb), err2mod)
    dtpn = err + linear_rgb
    return desaturate_linearRGB(dtpn)


def linearRGB_from_sRGB(im):
    """Convert sRGB to linearRGB, removing the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input sRGB image, normalized between [0,1]

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output RGB image
    """
    # Ensure im is a NumPy array and normalized between [0,1]
    im = np.array(im).astype(np.float32) / 255.0  # Convert to float and normalize
    out = np.zeros_like(im)
    small_mask = im < 0.04045
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] / 12.92
    out[large_mask] = np.power((im[large_mask] + 0.055) / 1.055, 2.4)
    return out

def sRGB_from_linearRGB(im):
    """Convert linearRGB to sRGB, applying the gamma correction.

    Formula taken from Wikipedia https://en.wikipedia.org/wiki/SRGB

    Parameters
    ==========
    im : array of shape (M,N,3) with dtype float
        The input RGB image, normalized between [0,1].
        It will be clipped to [0,1] to avoid numerical issues with gamma.

    Returns
    =======
    im : array of shape (M,N,3) with dtype float
        The output sRGB image
    """
    out = np.zeros_like(im)
    # Make sure we're in range, otherwise gamma will go crazy.
    im = np.clip(im, 0., 1.)
    small_mask = im < 0.0031308
    large_mask = np.logical_not(small_mask)
    out[small_mask] = im[small_mask] * 12.92
    out[large_mask] = np.power(im[large_mask], 1.0 / 2.4) * 1.055 - 0.055
    return out

def severity(score, max_score=100):
    """
    Calculate the severity factor based on the given score.
    Here, the score is assumed to be a percentage, where 100 represents normal.
    If your cone scores are non-linear, consider adjusting the formula.
    """
    return (1 - float(score / max_score))


def process_image_simulate(image_path, prognosis, *generic_scores):
    """Adjust image based on user-specific prognosis"""

    print("Received generic scores:", generic_scores)  # Debugging
    print(f"Prognosis: {prognosis}")

    if len(generic_scores) != 3:  # Ensure there are exactly 3 scores
        raise ValueError("Expected 3 scores, got {}".format(len(generic_scores)))

    image = Image.open(image_path)

    print("Image mode before splitting:", image.mode)  # Debugging

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Custom logic for user prognosis
    if prognosis == "tritan":
        linear_rgb = linearRGB_from_sRGB(image)
        sim_cvd_linear_rgb = simulate_tritan(linear_rgb)
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        sim_cvd_sRGB = np.clip( sim_cvd_sRGB * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(sim_cvd_sRGB)

    elif prognosis == "deutan":
        print(f"Begining processing for {prognosis}")
        sev = severity(generic_scores[1], 100)
        print(f"Severity: {sev}")
        linear_rgb = linearRGB_from_sRGB(image)
        print("Converted from sRGB to linear RGB")
        sim_cvd_linear_rgb = simulate(linear_rgb, prognosis, sev)
        print("Successful simulation")
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        print("Converted from linear RGB to sRGB")
        sim_cvd_sRGB = np.clip( sim_cvd_sRGB * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(sim_cvd_sRGB)

    elif prognosis == "protan":
        print(f"Begining processing for {prognosis}")
        sev = severity(generic_scores[0], 100)
        print(f"Severity: {sev}")
        linear_rgb = linearRGB_from_sRGB(image)
        print("Converted from sRGB to linear RGB")
        sim_cvd_linear_rgb = simulate(linear_rgb, prognosis, sev)
        print("Successful simulation")
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        print("Converted from linear RGB to sRGB")
        sim_cvd_sRGB = np.clip(sim_cvd_sRGB * 255, 0, 255).astype(np.uint8) # Check this!
        return Image.fromarray(sim_cvd_sRGB)

    return sim_cvd_sRGB


def process_image_daltonize(image_path, prognosis, *generic_scores):
    """Adjust image based on user-specific prognosis"""

    print("Received generic scores:", generic_scores)  # Debugging

    if len(generic_scores) != 3:  # Ensure there are exactly 3 scores
        raise ValueError("Expected 3 scores, got {}".format(len(generic_scores)))

    image = Image.open(image_path)

    print("Image mode before splitting:", image.mode)  # Debugging

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Custom logic for user prognosis
    if prognosis == "tritan":
        linear_rgb = linearRGB_from_sRGB(image)
        sim_cvd_linear_rgb = daltonize(linear_rgb, prognosis)
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        sim_cvd_sRGB = np.clip( sim_cvd_sRGB * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(sim_cvd_sRGB)

    elif prognosis == "deutan":
        print(f"Begining processing for {prognosis}")
        sev = severity(generic_scores[1], 100)
        print(f"Severity: {sev}")
        linear_rgb = linearRGB_from_sRGB(image)
        print("Converted from sRGB to linear RGB")
        sim_cvd_linear_rgb = daltonize(linear_rgb, prognosis, sev)
        print("Successful simulation")
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        print("Converted from linear RGB to sRGB")
        sim_cvd_sRGB = np.clip( sim_cvd_sRGB * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(sim_cvd_sRGB)

    elif prognosis == "protan":
        print(f"Begining processing for {prognosis}")
        sev = severity(generic_scores[0], 100)
        print(f"Severity: {sev}")
        linear_rgb = linearRGB_from_sRGB(image)
        print("Converted from sRGB to linear RGB")
        sim_cvd_linear_rgb = daltonize(linear_rgb, prognosis, sev)
        print("Successful simulation")
        sim_cvd_sRGB = sRGB_from_linearRGB(sim_cvd_linear_rgb)
        print("Converted from linear RGB to sRGB")
        sim_cvd_sRGB = np.clip(sim_cvd_sRGB * 255, 0, 255).astype(np.uint8) # Check this!
        return Image.fromarray(sim_cvd_sRGB)

    return sim_cvd_sRGB

