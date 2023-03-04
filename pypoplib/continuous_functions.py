import numpy as np

import pypoplib.base_functions as base_functions
from pypoplib.shifted_functions import _load_shift_vector
from pypoplib.rotated_functions import _load_rotation_matrix


# helper functions
def load_shift_and_rotation(func, x, shift_vector=None, rotation_matrix=None):
    shift_vector = _load_shift_vector(func, x, shift_vector)
    rotation_matrix = _load_rotation_matrix(func, x, rotation_matrix)
    return shift_vector, rotation_matrix


def sphere(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(sphere, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.sphere(x)
    return y


def cigar(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(cigar, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.cigar(x)
    return y


def discus(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(discus, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.discus(x)
    return y


def cigar_discus(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(cigar_discus, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.cigar_discus(x)
    return y


def ellipsoid(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(ellipsoid, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.ellipsoid(x)
    return y


def different_powers(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(different_powers, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.different_powers(x)
    return y


def schwefel221(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(schwefel221, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel221(x)
    return y


def step(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(step, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.step(x)
    return y


def rosenbrock(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(rosenbrock, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.rosenbrock(x)
    return y


def schwefel12(x, shift_vector=None, rotation_matrix=None):
    shift_vector, rotation_matrix = load_shift_and_rotation(schwefel12, x, shift_vector, rotation_matrix)
    x = np.dot(rotation_matrix, x - shift_vector)
    y = base_functions.schwefel12(x)
    return y
