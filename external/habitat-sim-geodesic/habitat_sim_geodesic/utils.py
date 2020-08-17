#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import List, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import quaternion


def quat_from_coeffs(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from the coeffs returned by the simulator backend

    :param coeffs: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
        where :math:`q = a + bi + cj + dk`
    :return: A quaternion from the coeffs
    """
    quat = np.quaternion(1, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quat_to_coeffs(quat: np.quaternion) -> np.ndarray:
    r"""Converts a quaternion into the coeffs format the backend expects

    :param quat: The quaternion
    :return: Coefficients of a quaternion in :py:`[b, c, d, a]` format,
        where :math:`q = a + bi + cj + dk`
    """
    coeffs = np.empty(4)
    coeffs[0:3] = quat.imag
    coeffs[3] = quat.real
    return coeffs


def quat_to_angle_axis(quat: np.quaternion) -> Tuple[float, np.ndarray]:
    r"""Converts a quaternion to angle axis format

    :param quat: The quaternion
    :return:
        -   `float` --- The angle to rotate about the axis by
        -   `numpy.ndarray` --- The axis to rotate about. If :math:`\theta = 0`,
            then this is harded coded to be the +x axis
    """

    rot_vec = quaternion.as_rotation_vector(quat)

    theta = np.linalg.norm(rot_vec)
    if np.abs(theta) < 1e-5:
        w = np.array([1, 0, 0])
        theta = 0.0
    else:
        w = rot_vec / theta

    return (theta, w)


def quat_from_angle_axis(theta: float, axis: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion from angle axis format

    :param theta: The angle to rotate about the axis by
    :param axis: The axis to rotate about
    :return: The quaternion
    """
    axis = axis.astype(np.float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)


def quat_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.quaternion:
    r"""Creates a quaternion that rotates the first vector onto the second vector

    :param v0: The starting vector, does not need to be a unit vector
    :param v1: The end vector, does not need to be a unit vector
    :return: The quaternion

    Calculates the quaternion q such that

    .. code:: py

        v1 = quat_rotate_vector(q, v0)
    """

    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    if c < (-1 + 1e-8):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh[2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return np.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return np.quaternion(s * 0.5, *(axis / s))


def angle_between_quats(q1: np.quaternion, q2: np.quaternion) -> float:
    r"""Computes the angular distance between two quaternions

    :return: The angular distance between q1 and q2 in radians
    """

    q1_inv = np.conjugate(q1)
    dq = q1_inv * q2

    return 2 * np.arctan2(np.linalg.norm(dq.imag), np.abs(dq.real))


def quat_rotate_vector(q: np.quaternion, v: np.ndarray) -> np.ndarray:
    r"""Helper function to rotate a vector by a quaternion

    :param q: The quaternion to rotate the vector with
    :param v: The vector to rotate
    :return: The rotated vector

    Does

    .. code:: py

        v = (q * np.quaternion(0, *v) * q.inverse()).imag
    """

    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (q * vq * q.inverse()).imag


def habitat_to_mp3d(pt_habitat: np.ndarray) -> np.ndarray:
    return quat_rotate_vector(
        quat_from_two_vectors(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        pt_habitat,
    )


def mp3d_to_habitat(pt_mp3d: np.ndarray) -> np.ndarray:
    return quat_rotate_vector(
        quat_from_two_vectors(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        pt_mp3d,
    )
