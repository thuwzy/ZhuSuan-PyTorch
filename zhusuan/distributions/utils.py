# -*- coding: utf-8 -*-
import torch

floating_dtypes = (torch.float32, torch.float16, torch.float64)
log_floating_dtypes = (torch.float32, torch.float16, torch.float64)
integer_dtypes = (torch.int32, torch.int16, torch.float64)


def assert_same_dtype_in(tensors_with_name, dtypes=None):
    """
    Whether all types of tensors in `tensors_with_name` are the same and in the
    allowed `dtypes`.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :param dtypes: A list of allowed dtypes. If `None`, then all dtypes are
        allowed.

    :return: The dtype of `tensors`.
    """
    dtypes_set = set(dtypes) if dtypes else None
    expected_dtype = None
    for tensor, tensor_name in tensors_with_name:
        if dtypes_set and (tensor.dtype not in dtypes_set):
            if len(dtypes) == 1:
                raise TypeError(
                    '{}({}) must have dtype {}.'.format(
                        tensor_name, tensor.dtype, dtypes[0]))
            else:
                raise TypeError(
                    '{}({}) must have a dtype in {}.'.format(
                        tensor_name, tensor.dtype, dtypes))
        if not expected_dtype:
            expected_dtype = tensor.dtype
        elif expected_dtype != tensor.dtype:
            tensor0, tensor0_name = tensors_with_name[0]
            raise TypeError(
                '{}({}) must have the same dtype as {}({}).'.format(
                    tensor_name, tensor.dtype,
                    tensor0_name, tensor0.dtype))

    return expected_dtype


def assert_same_float_dtype(tensors_with_name):
    """
    Whether all tensors in `tensors_with_name` have the same floating type.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :return: The type of `tensors`.
    """
    return assert_same_dtype_in(tensors_with_name, floating_dtypes)

def assert_same_log_float_dtype(tensors_with_name):
    """
    Whether all tensors in `tensors_with_name` have the same floating type, which also support log/exp operations.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :return: The type of `tensors`.
    """
    return assert_same_dtype_in(tensors_with_name, log_floating_dtypes)