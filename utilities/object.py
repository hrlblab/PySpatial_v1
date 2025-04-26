import numpy


def downsample_labels(labels):
    """Convert a labels matrix to the smallest possible integer format"""
    labels_max = numpy.max(labels)
    if labels_max < 128:
        return labels.astype(numpy.int8)
    elif labels_max < 32768:
        return labels.astype(numpy.int16)
    return labels.astype(numpy.int32)


def crop_labels_and_image(labels, image):
    """Crop a labels matrix and an image to the lowest common size

    labels - a n x m labels matrix
    image - a 2-d or 3-d image

    Assumes that points outside of the common boundary should be masked.
    """
    min_dim1 = min(labels.shape[0], image.shape[0])
    min_dim2 = min(labels.shape[1], image.shape[1])

    if labels.ndim == 3:  # volume
        min_dim3 = min(labels.shape[2], image.shape[2])

        if image.ndim == 4:  # multichannel volume
            return (
                labels[:min_dim1, :min_dim2, :min_dim3],
                image[:min_dim1, :min_dim2, :min_dim3, :],
            )

        return (
            labels[:min_dim1, :min_dim2, :min_dim3],
            image[:min_dim1, :min_dim2, :min_dim3],
        )

    if image.ndim == 3:  # multichannel image
        return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2, :]

    return labels[:min_dim1, :min_dim2], image[:min_dim1, :min_dim2]


def size_similarly(labels, secondary):
    """Size the secondary matrix similarly to the labels matrix

    labels - labels matrix
    secondary - a secondary image or labels matrix which might be of
                different size.
    Return the resized secondary matrix and a mask indicating what portion
    of the secondary matrix is bogus (manufactured values).

    Either the mask is all ones or the result is a copy, so you can
    modify the output within the unmasked region w/o destroying the original.
    """
    if labels.shape[:2] == secondary.shape[:2]:
        return secondary, numpy.ones(secondary.shape, bool)
    if labels.shape[0] <= secondary.shape[0] and labels.shape[1] <= secondary.shape[1]:
        if secondary.ndim == 2:
            return (
                secondary[: labels.shape[0], : labels.shape[1]],
                numpy.ones(labels.shape, bool),
            )
        else:
            return (
                secondary[: labels.shape[0], : labels.shape[1], :],
                numpy.ones(labels.shape, bool),
            )

    #
    # Some portion of the secondary matrix does not cover the labels
    #
    result = numpy.zeros(
        list(labels.shape) + list(secondary.shape[2:]), secondary.dtype
    )
    i_max = min(secondary.shape[0], labels.shape[0])
    j_max = min(secondary.shape[1], labels.shape[1])
    if secondary.ndim == 2:
        result[:i_max, :j_max] = secondary[:i_max, :j_max]
    else:
        result[:i_max, :j_max, :] = secondary[:i_max, :j_max, :]
    mask = numpy.zeros(labels.shape, bool)
    mask[:i_max, :j_max] = 1
    return result, mask

