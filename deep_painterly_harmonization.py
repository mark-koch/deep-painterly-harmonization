"""
Tensorflow Implementation of "Deep Painterly Harmonization", Luan et al.

See: https://arxiv.org/abs/1804.03189
"""

import argparse
import cv2
import scipy.io
import sklearn.metrics.pairwise
import sys
import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Avoids initial logging messages
import tensorflow as tf


parser = argparse.ArgumentParser(description='Performs Deep Painterly Harmonization.')

parser.add_argument('-c', '--content', dest='content_img', required=True,
                    help='Content image (contains the inserted objects)')
parser.add_argument('-s', '--style', dest='style_img', required=True,
                    help='Original style image')
parser.add_argument('-o', '--out', dest='output_path', required=True,
                    help='Where to save the image')
parser.add_argument('-m', '--mask', dest='mask_img', default=None,
                    help='Mask image (a black image with the inserted objects marked in white). If not provided, the '
                         'program generates one automatically')

parser.add_argument('-a', '--art_stylization', dest='art_stylization', default=5, type=float,
                    help='Number representing the stylization in the style image. Think of a number from 1.0 (Baroque, '
                         'Renaissance, etc.) to 10.0 (Cubism, Expressionism, etc.). The value will be used to tune the '
                         'losses in pass 2.')
parser.add_argument('--vgg', dest='vgg_path', default='imagenet-vgg-verydeep-19.mat',
                    help='Path to the VGG-19 model file.')
parser.add_argument('--iters_p1', dest='iterations_pass1', default=1000, type=int,
                    help='Number of iterations to run the first pass')
parser.add_argument('--iters_p2', dest='iterations_pass2', default=1000, type=int,
                    help='Number of iterations to run the second pass')
parser.add_argument('--max_size', dest='max_size', default=700, type=int,
                    help='Maximum height or width of the generated image')
parser.add_argument('--print_loss', dest='print_loss', default=0, type=int,
                    help='Set to value greater than 0 to print loss every x interations instead of showing the progress'
                         'bar.')
parser.add_argument('--only', dest='only', choices=['pass1', 'pass2', 'post'], nargs='*',
                    default=['pass1', 'pass2', 'post'],
                    help='Only run part of the pipeline. Note that you can provide --inter_res to use as the result of'
                         'the skipped phases.')
parser.add_argument('--inter_res', dest='inter_res',
                    help='If pass 1 is skipped, this image will be used as a starting point for pass 2. If you only '
                         'want to run te postprocessing, this argument is required and will be used as input.')
parser.add_argument('--num_cores', dest='num_cores', default=4, type=int,
                    help='Set if you want to skip the postprocessing.')


parser.add_argument('--p1_content_weight', dest='pass1_content_weight', type=float, default=5,
                    help='Content loss weight for the first pass.')
parser.add_argument('--p1_style_weight', dest='pass1_style_weight', type=float, default=100,
                    help='Style loss weight for the first pass.')
parser.add_argument('--p1_tv_weight', dest='pass1_tv_weight', type=float, default=1e-3,
                    help='Total variance loss weight for the first pass.')
parser.add_argument('--p1_content_layers', dest='pass1_content_layers', nargs='*', default=['conv4_1'],
                    help='VGG layers to use to compute the content loss in pass 1.')
parser.add_argument('--p1_style_layers', dest='pass1_style_layers', nargs='*',
                    default=['conv3_1', 'conv4_1', 'conv5_1'],
                    help='VGG layers to use to compute the style loss in pass 1.')
parser.add_argument('--p1_content_layer_weights', dest='pass1_content_layer_weights', type=float, default=None,
                    help='Weight of each layer in the content loss of pass 1. If not provided, all layers are weighted '
                         'uniformely.')
parser.add_argument('--p1_style_layer_weights', dest='pass1_style_layer_weights', type=float, nargs='*', default=None,
                    help='Weight of each layer in the style loss of pass 1. If not provided, all layers are weighted '
                         'uniformely.')
parser.add_argument('--p1_learning_rate', dest='pass1_learning_rate', type=float, default=0.1,
                    help='Learning rate to use for pass 1 optimization.')

parser.add_argument('--p2_content_weight', dest='pass2_content_weight', type=float, default=None,
                    help='Content loss weight for the second pass.')
parser.add_argument('--p2_style_weight', dest='pass2_style_weight', type=float, default=None,
                    help='Style loss weight for the second pass.')
parser.add_argument('--p2_tv_weight', dest='pass2_tv_weight', type=float, default=None,
                    help='Total variance loss weight for the second pass. If not provided the loss will be set '
                         'dependent on the noise in the image.')
parser.add_argument('--p2_hist_weight', dest='pass2_hist_weight', type=float, default=None,
                    help='Histogram loss weight for the second pass.')
parser.add_argument('--p2_content_layers', dest='pass2_content_layers', nargs='*', default=['conv4_1'],
                    help='VGG layers to use to compute the content loss in pass 2.')
parser.add_argument('--p2_style_layers', dest='pass2_style_layers', nargs='*',
                    default=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
                    help='VGG layers to use to compute the style loss in pass 2.')
parser.add_argument('--p2_reference_layer', dest='pass2_reference_layer', default='conv4_1',
                    help='Reference layer for the consistent style remappin in pass 2.')
parser.add_argument('--p2_hist_layers', dest='pass2_hist_layers', nargs='*', default=['conv1_1', 'conv4_1'],
                    help='VGG layers to use to compute the histogram loss in pass 2.')
parser.add_argument('--p2_content_layer_weights', dest='pass2_content_layer_weights', type=float, nargs='*',
                    default=None,
                    help='Weight of each layer in the content loss of pass 2. If not provided, all layers are weighted '
                         'uniformely.')
parser.add_argument('--p2_style_layer_weights', dest='pass2_style_layer_weights', type=float, nargs='*', default=None,
                    help='Weight of each layer in the style loss of pass 2. If not provided, all layers are weighted '
                         'uniformely.')
parser.add_argument('--p2_hist_layer_weights', dest='pass2_hist_layer_weights', type=float, nargs='*', default=None,
                    help='Weight of each layer in the histogram loss of pass 2. If not provided, all layers are '
                         'weighted uniformely.')
parser.add_argument('--p2_learning_rate', dest='pass2_learning_rate', type=float, default=0.1,
                    help='Learning rate to use for pass 2 optimization.')


VGG_MEAN_VALUES = np.array([103.939, 123.68, 116.779]).reshape((1, 1, 1, 3))


def build_model(input_img, vgg_path):
    print('Loading VGG-19 model...')

    net = {}
    vgg = scipy.io.loadmat(vgg_path)
    vgg_layers = vgg['layers'][0]

    # Create Layers
    def conv_layer(layer_input, weights):
        conv = tf.nn.conv2d(layer_input, weights, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def relu_layer(layer_input, bias):
        relu = tf.nn.relu(layer_input + bias)
        return relu

    def pool_layer(layer_input):
        pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    # Get weights and biases of Layer i from the VGG model
    def get_weights(i):
        weights = vgg_layers[i][0][0][2][0][0]
        return tf.constant(weights)

    def get_bias(i):
        bias = vgg_layers[i][0][0][2][0][1]
        return tf.constant(np.reshape(bias, bias.size))

    # Build network:

    _, height, width, dimension = input_img.shape
    net['input'] = tf.Variable(np.zeros((1, height, width, dimension), dtype=np.float32))

    # Block 1
    net['conv1_1'] = conv_layer(net['input'], get_weights(0))
    net['relu1_1'] = relu_layer(net['conv1_1'], get_bias(0))

    net['conv1_2'] = conv_layer(net['relu1_1'], get_weights(2))
    net['relu1_2'] = relu_layer(net['conv1_2'], get_bias(2))

    net['pool1'] = pool_layer(net['relu1_2'])

    # Block 2
    net['conv2_1'] = conv_layer(net['pool1'], get_weights(5))
    net['relu2_1'] = relu_layer(net['conv2_1'], get_bias(5))

    net['conv2_2'] = conv_layer(net['relu2_1'], get_weights(7))
    net['relu2_2'] = relu_layer(net['conv2_2'], get_bias(7))

    net['pool2'] = pool_layer(net['relu2_2'])

    # Block 3
    net['conv3_1'] = conv_layer(net['pool2'], get_weights(10))
    net['relu3_1'] = relu_layer(net['conv3_1'], get_bias(10))

    net['conv3_2'] = conv_layer(net['relu3_1'], get_weights(12))
    net['relu3_2'] = relu_layer(net['conv3_2'], get_bias(12))

    net['conv3_3'] = conv_layer(net['relu3_2'], get_weights(14))
    net['relu3_3'] = relu_layer(net['conv3_3'], get_bias(14))

    net['conv3_4'] = conv_layer(net['relu3_3'], get_weights(16))
    net['relu3_4'] = relu_layer(net['conv3_4'], get_bias(16))

    net['pool3'] = pool_layer(net['relu3_4'])

    # Block 4
    net['conv4_1'] = conv_layer(net['pool3'], get_weights(19))
    net['relu4_1'] = relu_layer(net['conv4_1'], get_bias(19))

    net['conv4_2'] = conv_layer(net['relu4_1'], get_weights(21))
    net['relu4_2'] = relu_layer(net['conv4_2'], get_bias(21))

    net['conv4_3'] = conv_layer(net['relu4_2'], get_weights(23))
    net['relu4_3'] = relu_layer(net['conv4_3'], get_bias(23))

    net['conv4_4'] = conv_layer(net['relu4_3'], get_weights(25))
    net['relu4_4'] = relu_layer(net['conv4_4'], get_bias(25))

    net['pool4'] = pool_layer(net['relu4_4'])

    # Block 5
    net['conv5_1'] = conv_layer(net['pool4'], get_weights(28))
    net['relu5_1'] = relu_layer(net['conv5_1'], get_bias(28))

    net['conv5_2'] = conv_layer(net['relu5_1'], get_weights(30))
    net['relu5_2'] = relu_layer(net['conv5_2'], get_bias(30))

    net['conv5_3'] = conv_layer(net['relu5_2'], get_weights(32))
    net['relu5_3'] = relu_layer(net['conv5_3'], get_bias(32))

    net['conv5_4'] = conv_layer(net['relu5_3'], get_weights(34))
    net['relu5_4'] = relu_layer(net['conv5_4'], get_bias(34))

    net['pool5'] = pool_layer(net['relu5_4'])

    return net


def build_mask(mask_img, net):
    # The original mask needs to be transformed to the corresponding size for each layer. In the paper, the authors
    # resize the image, but in the original implementation, they half the resolution at each max pooling and use a
    # 3x3 average pooling (stride 1, padding 1) at each convolution layer in VGG. We use the latter approach:

    # def pool(x): return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    def pool(x): return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    def downscale_to(x, layer): return tf.image.resize_images(x, size=[layer.shape[1], layer.shape[2]],
                                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    mask = {}
    h, w = mask_img.shape
    mask['input'] = tf.constant(mask_img.reshape([1, h, w, 1]), dtype=np.float32)

    # Block 1
    mask['conv1_1'] = pool(mask['input'])
    mask['conv1_2'] = pool(mask['conv1_1'])
    mask['pool1'] = downscale_to(mask['conv1_2'], net['pool1'])
    # Block 2
    mask['conv2_1'] = pool(mask['pool1'])
    mask['conv2_2'] = pool(mask['conv2_1'])
    mask['pool2'] = downscale_to(mask['conv2_2'], net['pool2'])
    # Block 3
    mask['conv3_1'] = pool(mask['pool2'])
    mask['conv3_2'] = pool(mask['conv3_1'])
    mask['conv3_3'] = pool(mask['conv3_2'])
    mask['conv3_4'] = pool(mask['conv3_3'])
    mask['pool3'] = downscale_to(mask['conv3_4'], net['pool3'])
    # Block 4
    mask['conv4_1'] = pool(mask['pool3'])
    mask['conv4_2'] = pool(mask['conv4_1'])
    mask['conv4_3'] = pool(mask['conv4_2'])
    mask['conv4_4'] = pool(mask['conv4_3'])
    mask['pool4'] = downscale_to(mask['conv4_4'], net['pool4'])
    # Block 5
    mask['conv5_1'] = pool(mask['pool4'])
    mask['conv5_2'] = pool(mask['conv5_1'])
    mask['conv5_3'] = pool(mask['conv5_2'])
    mask['conv5_4'] = pool(mask['conv5_3'])
    mask['pool5'] = downscale_to(mask['conv5_4'], net['pool5'])

    return mask


def rearrange_style_activations_pass1(style_activation_layers, content_activation_layers):
    """ Performs pattern matching for the pass 1 reconstruction.

    Args:
        style_activation_layers: A Dictionary mapping layer names to the style image output of the corresponding layer.
            Pattern matching is going to performed for each layer in this dictionary.

        content_activation_layers: A Dictionary mapping layer names to the content image output of the corresponding
            layer. Note that the dictionary must contain all the keys that are given in 'style_activation_layers'.

    Returns:
        rearr_style_activation_layers: A dictionary with the pattern matched style layers.
        mapping: A dictionary specifying the performed mapping from the original to the rearranged style outputs for
            each layer.

    To reconstruct the style of the image, we want to consider the area of the original image, that already looks the
    most like the inserted object. Therfore we rearrange each pixel of the style image activations so that it coresponds
    to the most similar area of the content image.
    We consider a 3x3 patch around each pixel of the content layer activations inside the mask and find the most similar
    patch in the style layer activations. The pixel at the centre of this patch is used as the style target for the con-
    tent pixel.
    """
    # TODO only do the process for content pixels that are actually in the mask
    print("Perform independent mapping:")

    def get_patches(layer_activations):
        # tf.extract_image_patches returns [batch, h, w, 9 * filters]. The patch is flattened to 3x3=9 in the last di-
        # mension, but the filters are also in the last dimension
        ps = tf.image.extract_image_patches(layer_activations,
                                            ksizes=[1, 3, 3, 1],       # Patches have size 3x3
                                            strides=[1, 1, 1, 1],      # Patch centres are directly side by side
                                            rates=[1, 1, 1, 1],        # Don't "skip" pixels inside patch
                                            padding='SAME')\
            .eval()  # Execute directly
        ps = np.array(ps)
        # Remove batch dimension (it's alsway 1) and flatten xy to single axis. We want shape [patch=h*w, 9 * filters]
        ps = ps.reshape((ps.shape[1] * ps.shape[2], -1))
        return ps

    # Save mapping for each layer
    mapping = {}
    rearr_style_activation_layers = {}

    for layer in style_activation_layers:
        print("    Working on layer {}...".format(layer))
        _, h, w, _ = style_activation_layers[layer].shape
        style_patches = get_patches(style_activation_layers[layer])      # [h*w, 9*f]
        content_patches = get_patches(content_activation_layers[layer])  # [h*w, 9*f]
        # Calculate cosine distances between patches. Sklearn returns a matrix where dist[i,j] is the distance between
        # the i-th content patch and the j-th style patch. Resulting shape is [h*w, h*w]
        dist = sklearn.metrics.pairwise.pairwise_distances(content_patches, style_patches, metric='cosine')
        # For each content patch location h*w, find index h*w of corresponding style patch with minimal cosine distance
        idx = np.argmin(dist, axis=1)  # [h*w]
        idx = idx.reshape(h, w)  # Seperate into coordinates: [h, w]
        # Currently, style coordinates inside idx are flattened as h*w. Translate them to (h,w) coordinates. Returns two
        # arrays ys and xs of shape [h,w] cooresponding to the y and x coordinates of the matched style patches
        ys, xs = np.unravel_index(idx, [h, w])
        # Rearrange style activations (make copy so that we don't change the original)
        rearr_style_activation_layers[layer] = style_activation_layers[layer].copy()
        rearr_style_activation_layers[layer][0] = style_activation_layers[layer][0][ys, xs]
        mapping[layer] = idx

    return rearr_style_activation_layers, mapping


def gram_matrix(x, area, depth):
    f = tf.reshape(x, (area, depth))
    gram = tf.matmul(tf.transpose(f), f)
    return gram


def content_loss_pass1(net, mask, target_act_layers):
    total_loss = 0.
    for layer, weight in zip(args.pass1_content_layers, args.pass1_content_layer_weights):
        # Expand mask dimensions
        _, h, w, _ = mask[layer].shape
        m = tf.reshape(mask[layer], [1, h, w, 1])
        # Apply mask
        act = tf.multiply(net[layer], m)  # Activations of generated image
        target_act = tf.multiply(target_act_layers[layer], m)  # Original activations of the content to be matched
        # Calculate mse loss.
        # Note that we only divide by the number of activations INSIDE the mask times the number of filters
        _, height, width, filters = act.shape
        loss = tf.reduce_sum(tf.pow(act - target_act, 2)) / (tf.reduce_sum(mask[layer]) * filters.value)
        total_loss += loss * weight
    return total_loss


def style_loss_pass1(net, mask, target_act_layers):
    total_loss = 0.
    for layer, weight in zip(args.pass1_style_layers, args.pass1_style_layer_weights):
        # Expand mask dimensions
        _, h, w, f = net[layer].shape
        m = tf.reshape(mask[layer], [1, h, w, 1])
        m = tf.stop_gradient(m)
        # Apply mask
        act = tf.multiply(net[layer], m)  # Activations of generated image
        target_act = tf.multiply(target_act_layers[layer], m)  # Rearranged activations of the style img to be matched
        # Calculate mse loss with gram matrices
        _, height, width, filters = act.shape
        a = height.value * width.value  # a = Area of the layer
        f = filters.value               # f = Filters count

        pixels = tf.reduce_sum(mask[layer])
        loss = tf.losses.mean_squared_error(gram_matrix(act, a, f) / pixels, gram_matrix(target_act, a, f) / pixels)
        total_loss += loss * weight
    return total_loss


def rearrange_style_activations_pass2(style_activation_layers, mapping_pass1, reference_layer, mask, style_img,
                                      radius=2):
    """ Performs pattern matching for the pass 1 reconstruction.

    Args:
        style_activation_layers: A dictionary mapping layer names to the style image output of the corresponding
            layer. Pattern matching according to the reference layer is going to performed for each layer in this dict-
            ionary. Note that these need to be the orignal, unmapped layer outputs and the given 'refernce_layer' needs
            to be included.

        mapping_pass1: A dictionary containing the mapping for the pattern matching performed for pass 1. Needs to con-
            tain the given 'reference_layer'.

        reference_layer: Name of layer used to compute the reference mapping that is also going to be used for all the
            other layers.

        mask: A dictionary mapping layer names to the corresponding mask

        style_img: The original style image

        radius: Size of the radius in which to look for neightboring patches.

    Returns:
        rearr_style_activation_layers: A dictionary with the pattern matched style layers.

        duplicate_free_mask: A copy of the given mask dictionary but features that were seleted more than once are mask-
            ed out.


    Unlike the first pass, where each layer is rearranged independently, we only compute the mapping for one reference
    layer and apply it to all layers.
    To reinforce spatial consistency, we try to assign style pixels to input pixels that come from the same region in
    the orginal style image:
    We start with the mapped style image from pass 1. To get the final mapping for each pixel, we look at each neighbor-
    ing pixel in the a given region, look up the original position in the style image it came from and get the opposite
    neighbor as a potential candidate:

                Original style patch:                        Patch were n was mapped to in pass 1:
                       n - -                                               - - -
                       - p -                                               - n -
                       - - -                                               - - q

           -> q has the same relative position to n as n has to p
           -> We add q as a potential candidate

    Among these candidates we choose the one that is closest to all the neighbors. Note that in the original paper the
    authors compared the candidates to the neigbors in the pass 1 rearranged layer, while in their implementation they
    comared them to the 'q'-values in the above illustration.
    """
    # TODO: This can probably be done more efficiently...
    print("Perform consistent mapping:")
    _, h, w, f = style_activation_layers[reference_layer].shape
    mapping = np.empty([h, w], dtype=(int, 2))

    print("    Mapping reference layer {}...".format(reference_layer))
    # Iterate over each pixel in the reference style layer
    for y in range(h):
        for x in range(w):
            candidates = []
            candidate_coords = []
            neighbors = []
            # Iterate over relative neighbor coordinates:
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # Check if neighbor is out of bounds
                    if h > y + i > 0 and w > x + j > 0:
                        # Get x,y coordinates of pixel this neighbor got mapped to during pass 1 and add
                        match_y, match_x = np.unravel_index(mapping_pass1[y + i, x + j], [h, w])
                        # In the paper this pixel was then added to the neighbor list:
                        # neighbors.append(style_activation_layers[reference_layer][0, match_y, match_x])
                        if h > match_y - i > 0 and w > match_x - j > 0:
                            # candidates.append(style_activation_layers[reference_layer][0, match_y - i, match_x - j])
                            candidates.append(style_img[match_y - i, match_x - j])
                            candidate_coords.append((match_y - i, match_x - j))
                            # Implementation of the authors (vs commented out line above):
                            # neighbors.append(style_activation_layers[reference_layer][0, match_y - i, match_x - j])
                            neighbors.append(style_img[match_y - i, match_x - j])
            if len(candidates) > 0:
                # Compute squared difference between the neigboring pixels and each candidate
                distances = ((np.array(candidates)[:, np.newaxis, :] - np.array(neighbors)[np.newaxis, :, :]) ** 2)\
                    .sum(axis=2).sum(axis=1)
                mapping[y, x] = candidate_coords[distances.argmin()]
            else:
                mapping[y, x] = (y, x)

    # Remove duplicates from mapping: duplicate entries are replaced by (-1, -1)
    used_mapping = np.zeros([h, w])
    for y in range(h):
        for x in range(w):
            if mask[args.pass2_reference_layer][0, y, x] > 0.1:
                my, mx = mapping[y, x]
                if used_mapping[my, mx] != 0:
                    mapping[y, x] = (-1, -1)
                else:
                    used_mapping[my, mx] = 1

    # Now apply this mapping to each layer (scale accordingly)
    rearr_style_activation_layer = {}
    duplicate_free_mask = {}

    for layer in mask:
        duplicate_free_mask[layer] = mask[layer].copy()
    for layer in style_activation_layers:
        print("    Propagating to layer {}...".format(layer))
        rearr_style_activation_layer[layer] = style_activation_layers[layer].copy()
        _, new_h, new_w, _ = style_activation_layers[layer].shape
        scale_y, scale_x = h / float(new_h), w / float(new_w)
        for y in range(new_h):
            # Relative y coordinate in reference layer
            ry = min(int(round(y*scale_y)), h - 1)
            for x in range(new_w):
                if mask[layer][0, y, x] > 0.1:
                    # Relative x coordinate in reference layer
                    rx = min(int(round(x*scale_x)), w - 1)
                    # Get matching pixel in the reference layer
                    m_ry, m_rx = mapping[ry, rx]
                    if m_ry != -1 or m_rx != -1:  # Duplicate mappings are marked by -1
                        # Translate back to position in current layer
                        m_y = min(int(round(y + (m_ry-ry)/scale_y)), new_h-1)
                        m_x = min(int(round(x + (m_rx-rx)/scale_x)), new_w-1)
                        rearr_style_activation_layer[layer][0, y, x] = style_activation_layers[layer][0, m_y, m_x]
                    else:
                        duplicate_free_mask[layer][0, y, x] = 0

    return rearr_style_activation_layer, duplicate_free_mask


def content_loss_pass2(net, mask, target_act_layers):
    total_loss = 0.
    for layer, weight in zip(args.pass2_content_layers, args.pass2_content_layer_weights):
        # Expand mask dimensions
        _, h, w, f = net[layer].shape
        m = tf.stack([mask[layer].reshape([h, w]) for _ in range(f.value)], axis=2)
        m = tf.stack(m, axis=0)
        m = tf.expand_dims(m, 0)
        # Apply mask
        act = tf.multiply(net[layer], m)  # Activations of generated image
        target_act = tf.multiply(target_act_layers[layer], m)  # Original activations of the content to be matched
        # Calculate mse loss.
        # Note that we only divide by the number of activations INSIDE the mask times the number of filters
        _, height, width, filters = act.shape
        loss = tf.reduce_sum(tf.pow(act - target_act, 2)) / (tf.reduce_sum(mask[layer]) * filters.value)
        total_loss += loss * weight
    return total_loss


def style_loss_pass2(net, mask, target_mask, target_act_layers):
    """  Generates the style loss for pass 2.

    This is slightly different to pass 1: The neirest neighbor sampling used for the style mapping in pass 2 may select
    the same patches multiple times. But we want these patches to contribute only once to the style loss computation.
    Thus we apply a special mask to target layers: This 'target_mask' not only contains the parts of the regular mask,
    but should also remove additional occurrences of these patches.
    """
    total_loss = 0.
    for layer, weight in zip(args.pass2_style_layers, args.pass2_style_layer_weights):
        # Expand mask dimensions
        _, h, w, f = net[layer].shape
        m = tf.reshape(mask[layer], [1, h, w, 1])
        target_m = tf.reshape(target_mask[layer], [1, h, w, 1])
        # Apply mask
        act = tf.multiply(net[layer], m)
        target_act = tf.multiply(target_act_layers[layer], target_m)  # Use duplicate removing mask here
        # Because more pixels where masked out in target_act, we need to normalize. As the gram matrix multiplies the
        # values, we need to use the square root. (Thanks to www.sgugger.github.io/deep-painterly-harmonization.html)
        # TODO: Commented out for now, we follow the authors
        # target_act = target_act * tf.sqrt(tf.reduce_sum(m) / tf.reduce_sum(target_m))
        _, height, width, filters = act.shape
        a = height.value * width.value  # a = Area of the layer
        f = filters.value               # f = Filters count
        loss = tf.losses.mean_squared_error(gram_matrix(act, a, f) / tf.reduce_sum(mask[layer]),
                                            gram_matrix(target_act, a, f) / tf.reduce_sum(target_mask[layer])) / f
        total_loss += loss * weight
    return total_loss


def histogram_loss_old(net, mask, target, target_mask):
    """ This version is way to slow. See improved version below. """
    total_loss = 0.
    for layer, weight in zip(args.pass2_hist_layers, args.pass2_hist_layer_weights):
        _, h, w, f = net[layer].shape
        # Mask the features, flatten them and remove entries outside the mask: [?, f]
        content_masked = tf.gather(tf.reshape(net[layer], [-1, f]),
                                   tf.squeeze(tf.where(tf.reshape(mask[layer], [-1]) > 0.1)))
        target_masked = tf.gather(tf.reshape(target[layer], [-1, f]),
                                  tf.squeeze(tf.where(tf.reshape(target_mask[layer], [-1]) > 0.1)))
        # Get range of values for each filter
        max_value = tf.reduce_max([tf.reduce_max(content_masked, axis=0), tf.reduce_max(target_masked, axis=0)], axis=0)
        min_value = tf.reduce_min([tf.reduce_min(content_masked, axis=0), tf.reduce_min(target_masked, axis=0)], axis=0)

        # Get histograms for each filter: [255, f]
        def get_hist(x): return tf.stack([tf.histogram_fixed_width(x[:, ff], [min_value[ff], max_value[ff]], nbins=255)
                                          for ff in range(f)], axis=1)
        hist_content = tf.to_float(get_hist(content_masked))
        hist_target = tf.to_float(get_hist(target_masked))
        # Normalize histograms (i.e. divide by the number of pixels to get a probability distribution), because we used
        # different masks for content and target
        hist_content = hist_content / tf.reduce_sum(mask[layer])
        hist_target = hist_target / tf.reduce_sum(target_mask[layer])
        # Compute cumulative histograms
        cum_hist_content = tf.cumsum(hist_content, axis=0)
        cum_hist_target = tf.cumsum(hist_target, axis=0)

        # For each bin in the content hist, find the bin in the target hist with the closest probability
        bin_matching = tf.stack([tf.map_fn(lambda bin_prob: tf.argmin(tf.abs(cum_hist_target[:, ff] - bin_prob)),
                                           cum_hist_content[:, ff], dtype=tf.int64, back_prop=False)
                                 for ff in range(f)], axis=1)
        # Determine the bin index for each pixel in the content layer
        bin_width = (max_value - min_value) / 255.0
        content_bin_idx = tf.clip_by_value(tf.to_int32(content_masked / bin_width), 0, 254)
        # Replace the color value of each pixel by the value of the matching target bin
        # TODO: This is not optimal. If mutliple pixel match the same bin, they all get the same value. It would be bet-
        #  ter to spread them evenly between the whole bin_width. This only gets worse if the value range is much great-
        #  er than 255.
        content_remapped = \
            tf.stack([min_value[ff] +
                      bin_width[ff]*tf.to_float(tf.gather(bin_matching[:, ff], content_bin_idx[:, ff]))+bin_width[ff]/2
                      for ff in range(f)], axis=1)

        # We cannot compute the gradient for the histogram remapping but it would be zero almost everywhere anyway, so
        # we treat it as a constant instead.
        content_remapped = tf.stop_gradient(content_remapped)
        total_loss += tf.losses.mean_squared_error(content_masked, content_remapped) * weight
    return total_loss


def histogram_loss(net, mask, target, target_mask, num_bins=255):
    """
    The histogram loss is computed by first transforming the filters of content layers so that their histograms match
    up with the histograms of the corresponding target filters. The loss is the mean squared difference between the
    layers and their hisogram remapped versions.

    See https://arxiv.org/pdf/1701.08893v2.pdf
    """
    total_loss = 0.
    for layer, weight in zip(args.pass2_hist_layers, args.pass2_hist_layer_weights):
        _, h, w, f = net[layer].shape
        # Mask the features, flatten them and remove entries outside the mask: [?, f]
        content_masked = tf.gather(tf.reshape(net[layer], [-1, f]),
                                   tf.squeeze(tf.where(tf.reshape(mask[layer], [-1]) > 0.1)))
        target_masked = tf.gather(tf.reshape(target[layer], [-1, f]),
                                  tf.squeeze(tf.where(tf.reshape(target_mask[layer], [-1]) > 0.1)))

        # Get range of values for each filters
        max_value = tf.reduce_max([tf.reduce_max(content_masked, axis=0), tf.reduce_max(target_masked, axis=0)], axis=0)
        min_value = tf.reduce_min([tf.reduce_min(content_masked, axis=0), tf.reduce_min(target_masked, axis=0)], axis=0)
        # Get histograms of target layer for each filter: [255, f]
        hist = tf.stack([tf.histogram_fixed_width(target_masked[:, ff], [min_value[ff], max_value[ff]], nbins=num_bins)
                         for ff in range(f)], axis=1)
        # Normalize histogram, because we use different masks for content and target
        hist = tf.to_float(hist) * (tf.reduce_sum(mask[layer]) / tf.reduce_sum(target_mask[layer]))
        # Compute cumulative histograms and normalize them (norm is needed because we masked out more target pixels)
        cum_hist = tf.to_float(tf.cumsum(hist, axis=0))
        # We also need a version of the histogram shifted one to the right, so that i-th value in this histogram corres-
        # ponds to the previous (i-1 th) value in the actual histogram
        prev_cum_hist = tf.concat([tf.zeros([1, f]), cum_hist[:-1, :]], axis=0)

        # For each index i and filter f in our flattened content layer, we need to find the first index idx such that
        # cum_hist[idx, f] > i. To do this efficiently we first create a tensor with all the indices 1,2,3,...
        rng = tf.range(0, tf.reduce_sum(mask[layer]), delta=1)
        # Then find all the indices j for which cum_hist[j, f] < i. Since cumulative histograms are always sorted, the
        # sum of all these indices j is the index idx we were looking for.
        smaller = tf.expand_dims(cum_hist, 0) - tf.reshape(rng, [-1, 1, 1])
        idx = tf.reduce_sum(tf.cast(smaller < 0, tf.int32), axis=1)
        idx = tf.clip_by_value(idx, 0, num_bins - 1)

        # Width of one bin
        step = (max_value - min_value) / num_bins

        # This would work:
        # content_remapped = min_value + tf.to_float(idx) * step
        # But if there were multiple pixels assigned to the same idx they would all get the same value. It would be bet-
        # ter to uniformly spread them across the whole idx bin. Therfore we do this instead:

        # Helper to access tensor with multidimensional index
        def get(x, i): return tf.stack([tf.gather(x[:, ff], i[:, ff]) for ff in range(f)], axis=1)
        ratio = (tf.to_float(idx) - get(prev_cum_hist, idx)) / (1e-8 + get(hist, idx))
        ratio = tf.clip_by_value(ratio, 0, 1)
        content_remapped = min_value + (ratio + tf.cast(idx, tf.float32)) * step

        # The values in 'content_remapped' are sorted, so we need to put them back into the original order. First calcu-
        # late the mapping that sorts the content layer, than apply the inverse of this mapping.
        sort_indices = tf.argsort(content_masked, axis=0)
        unsort_indices = tf.argsort(sort_indices, axis=0)
        content_remapped = get(content_remapped, unsort_indices)

        # We cannot compute the gradient for the histogram remapping but it would be zero almost everywhere anyway, so
        # we treat it as a constant instead.
        content_remapped = tf.stop_gradient(content_remapped)
        loss = tf.losses.mean_squared_error(content_masked, content_remapped)
        total_loss += loss
    return total_loss


def run_postprocessing(img, content_img, target_img, num_cores=4):
    print("Perform post processing:")
    # Trim batch dimension
    img = postprocess_img(img)
    content_img = postprocess_img(content_img)
    target_img = postprocess_img(target_img)
    # Apply guided filter. The authors state that they use the luminance channel as guide, but in their code they actu-
    # ally use the original content image
    print("    Guided filter...")
    from cv2.ximgproc import guidedFilter
    img_denoise = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    # img_denoise[:, :, 1] = guidedFilter(content_img, img_denoise[:, :, 1], radius=2, eps=0.01)
    # img_denoise[:, :, 2] = guidedFilter(content_img, img_denoise[:, :, 2], radius=2, eps=0.01)
    img_denoise[:, :, 1] = guidedFilter(img_denoise[:, :, 0], img_denoise[:, :, 1], radius=2, eps=0.01)
    img_denoise[:, :, 2] = guidedFilter(img_denoise[:, :, 0], img_denoise[:, :, 2], radius=2, eps=0.01)
    img_denoise = cv2.cvtColor(img_denoise, cv2.COLOR_LAB2BGR) / 255.0
    # Perform patch match to get base image
    print("    PatchMatch on {} cores...".format(num_cores))
    from patchmatch import patch_match_parallel, reconstruct_avg
    m = patch_match_parallel(
        img_denoise, target_img/255.0, iters=5, patch_size=7, alpha=0.5, w=None, num_cores=num_cores)
    img_base = reconstruct_avg(target_img/255.0, m, patch_size=7)
    # As the averaging above removed details from the image we want to reintroduce the details by further smoothing
    # img_denoise and adding the resulting difference. Note that the authors used a gaussian blur in their code as op-
    # posed to another guided filter in the paper.
    r = 3
    img_blur = cv2.GaussianBlur(img_denoise, (2*r+1, 2*r+1), r/3)
    img_detail = img_denoise - img_blur
    img_final = (img_base + img_detail) * 255
    # Return the image in the same format we revieved it
    return preprocess_img(img_final)


def generate_mask(content, target):
    """ Generates a mask image based on the differences between 'content' and 'target'. """
    subtractor = cv2.createBackgroundSubtractorMOG2()
    subtractor.apply(target)
    mask = subtractor.apply(content)
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    return mask


def resize_fit(img, max_size):
    """ Resizes an image if its height or width is greater than 'max_size'. """
    height, width = img.shape[0], img.shape[1]
    if height > width and height > max_size:
        width = (float(max_size) / float(height)) * width
        img = cv2.resize(img, dsize=(int(width), max_size), interpolation=cv2.INTER_AREA)
    if width > max_size:
        height = (float(max_size) / float(width)) * height
        img = cv2.resize(img, dsize=(max_size, int(height)), interpolation=cv2.INTER_AREA)
    return img


def preprocess_img(img):
    # Convert BGR to RGB
    img = img[..., ::-1]
    # Add batch dimension
    img = img[np.newaxis, :, :, :].astype(np.float)
    img -= VGG_MEAN_VALUES
    return img


def postprocess_img(img):
    img = np.copy(img)
    img += VGG_MEAN_VALUES
    # Remove batch dimension
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    # Convert RGB to BGR
    img = img[..., ::-1]
    return img


def main():
    start_time = time.time()

    # Disable info and warning logging
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Load images
    content_img = cv2.imread(args.content_img, cv2.IMREAD_COLOR)
    style_img = cv2.imread(args.style_img, cv2.IMREAD_COLOR)
    if args.mask_img is not None:
        mask_img = cv2.imread(args.mask_img, cv2.IMREAD_GRAYSCALE)
        assert mask_img.shape[0] == content_img.shape[0] and mask_img.shape[1] == content_img.shape[1], \
            "Mask image must have the same size as the content image!"
    else:
        assert content_img.shape[0] == style_img.shape[0] and content_img.shape[1] == style_img.shape[1], \
            "Cannot generate mask when content and style image have different sizes. Please provide custom a mask image"
        file, extension = os.path.splitext(args.output_path)
        print("Generating mask image and saving at " + file + "_mask" + extension)
        mask_img = generate_mask(content_img, style_img)
        cv2.imwrite(file + "_mask" + extension, postprocess_img(mask_img))
    style_img = resize_fit(style_img, args.max_size)
    style_img = preprocess_img(style_img)
    content_img = resize_fit(content_img, args.max_size)
    content_img = preprocess_img(content_img)
    mask_img = resize_fit(mask_img, args.max_size)
    mask_img = mask_img.astype(np.float32) / 255.0

    # Different cases for --only
    skip_pass1 = 'pass1' not in args.only
    skip_pass2 = 'pass2' not in args.only
    skip_post = 'post' not in args.only
    assert not (skip_pass1 and skip_pass2 and skip_post), "If you skip every stage there is nothing to do."
    # Intermedite result
    inter_res = None
    if args.inter_res is not None:
        inter_res = preprocess_img(cv2.imread(args.inter_res, cv2.IMREAD_COLOR))
    # If only post processing is done, inter_res is required
    if skip_pass1 and skip_pass2:
        assert inter_res is not None, "Specify --interres when using --only post"

    # Layer weights
    if args.pass1_content_layer_weights is None:
        args.pass1_content_layer_weights = [1.0/len(args.pass1_content_layers) for _ in args.pass1_content_layers]
    if args.pass1_style_layer_weights is None:
        args.pass1_style_layer_weights = [1.0/len(args.pass1_style_layers) for _ in args.pass1_style_layers]
    if args.pass2_content_layer_weights is None:
        args.pass2_content_layer_weights = [1.0 / len(args.pass2_content_layers) for _ in args.pass2_content_layers]
    if args.pass2_style_layer_weights is None:
        args.pass2_style_layer_weights = [1.0 / len(args.pass2_style_layers) for _ in args.pass2_style_layers]
    assert len(args.pass1_content_layer_weights) == len(args.pass1_content_layers), \
        "Number of pass 1 content layers does not match up with number of given weights!"
    assert len(args.pass1_style_layer_weights) == len(args.pass1_style_layers), \
        "Number of pass 1 style layers does not match up with number of given weights!"
    assert len(args.pass2_content_layer_weights) == len(args.pass2_content_layers), \
        "Number of pass 2 content layers does not match up with number of given weights!"
    assert len(args.pass2_style_layer_weights) == len(args.pass2_style_layers), \
        "Number of pass 2 style layers does not match up with number of given weights!"

    # Pass 2 loss weights
    if not skip_pass2:
        if args.pass2_content_weight is None:
            args.pass2_content_weight = 1
        if args.pass2_style_weight is None:
            args.pass2_style_weight = args.art_stylization
        if args.pass2_hist_weight is None:
            args.pass2_hist_weight = args.art_stylization
        if args.pass2_tv_weight is None:
            args.pass2_tv_weight = 1e-3  # TODO: Calculate based on image noise

    global _iter
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Preparation
        if not skip_pass1 or not skip_pass2:
            net = build_model(content_img, args.vgg_path)

            # Dilate Mask: Gaussion blur smoothly enlarges mask
            dilated_mask = cv2.GaussianBlur(mask_img, (35, 35), 35 / 3)
            dilated_mask[dilated_mask >= 0.1] = 1  # Apply treshold
            for i in range(0):
                dilated_mask = cv2.GaussianBlur(dilated_mask, (35, 35), 35 / 3)
                dilated_mask[dilated_mask >= 0.1] = 1

            mask = build_mask(dilated_mask, net)
            tight_mask = build_mask(mask_img, net)

            # Run mask in session to get concrete outputs
            mask = dict(zip(mask.keys(),  # Zip to create key value pairs of layers and masks
                            sess.run([mask[layer] for layer in mask.keys()])))
            tight_mask = dict(zip(tight_mask.keys(),
                              sess.run([tight_mask[layer] for layer in tight_mask.keys()])))
            # Generate original activations of style and content
            sess.run(net['input'].assign(content_img))
            content_activation_layers = dict(zip(net.keys(),  # Zip to create key value pairs of layers and acts
                                                 sess.run([net[layer] for layer in net.keys()])))
            sess.run(net['input'].assign(style_img))
            style_activation_layers = dict(zip(net.keys(),  # Zip to create key value pairs of layers and acts
                                               sess.run([net[layer] for layer in net.keys()])))

        # Pass 1
        if skip_pass1:
            print("Skipping pass 1")
            output_pass1 = inter_res if inter_res is not None else content_img
        else:
            # Rearrange activations in style layers according to patch mapping to content
            mapping_layers = set(args.pass1_style_layers + args.pass1_content_layers)
            style_activation_layers_pass1, _ = \
                rearrange_style_activations_pass1(dict([(l, style_activation_layers[l]) for l in mapping_layers]),
                                                  content_activation_layers)

            loss_c = content_loss_pass1(net, mask, content_activation_layers)
            loss_s = style_loss_pass1(net, mask, style_activation_layers_pass1)
            loss_tv = tf.image.total_variation(net['input'] * mask['input'])
            loss = args.pass1_content_weight*loss_c + args.pass1_style_weight*loss_s + args.pass1_tv_weight*loss_tv

            sess.run(tf.global_variables_initializer())
            sess.run(net['input'].assign(content_img))  # We begin with the unmodified content image

            print("Loading optimizer...")

            # We use the Scipy LBFGS optimizer. To get info about progress, we need to define a callback function
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                               options={'maxiter': args.iterations_pass1,
                                                                        'eps': args.pass1_learning_rate})

            def step_callback(_):
                global _iter
                if args.print_loss <= 0:
                    # Show progress bar
                    if _iter == 0:
                        print("")
                    percentage = (_iter+1) / float(args.iterations_pass1)
                    filled_length = int(round(50 * percentage))
                    bar = '█' * filled_length + '-' * (50 - filled_length)
                    sys.stdout.write("\rPass 1: |{}| {:0.1f}%".format(bar, percentage*100))
                    sys.stdout.flush()
                    if _iter == args.iterations_pass1-1:
                        print("\n")
                _iter += 1

            def loss_callback(l_c, l_s, l_tv):
                global _iter
                if args.print_loss > 0 and _iter % args.print_loss == 0:
                    print("{}: content={}, style={}, tv={} ".format(_iter, l_c, l_s, l_tv))

            _iter = 0
            optimizer.minimize(sess, step_callback=step_callback, loss_callback=loss_callback,
                               fetches=[loss_c, loss_s, loss_tv])
            generated_image = sess.run(net['input'])

            # Apply mask again to reconstruct image
            output_mask = cv2.GaussianBlur(mask_img, (3, 3), 1)  # Smooth edges
            output_mask = np.stack((output_mask, )*3, axis=-1)  # To RGB
            output_mask = np.expand_dims(output_mask, 0)  # Add batch dimension

            output_pass1 = generated_image * output_mask + style_img * (1 - output_mask)
            # Save image
            file, extension = os.path.splitext(args.output_path)
            cv2.imwrite(file + "_pass1" + extension, postprocess_img(output_pass1))

        # Pass 2
        if skip_pass2:
            print("Skipping pass 2")
            if not skip_pass1:
                output_pass2 = output_pass1
            elif not skip_post:
                output_pass2 = inter_res
        else:
            # Compute reference layer activation for pass 1 output as new input
            sess.run(net['input'].assign(output_pass1))
            content_activation_reference = sess.run(net[args.pass2_reference_layer])

            # Use the pass 1 output to recompute the mapping for the refernce layer. In their code the authors normalize
            # the features at this point. TODO: Decide what to do here
            def norm(x): return np.sqrt((x[0] * x[0]).sum(axis=0).sum(axis=0)) + 1e-8
            layer = args.pass2_reference_layer
            _, mapping_pass1 = rearrange_style_activations_pass1(
                {layer: style_activation_layers[layer] / norm(style_activation_layers[layer])},
                {layer: content_activation_reference / norm(content_activation_reference)})

            # Perform pass2 mapping. Note that in the paper the authors say they used the reference layer features for
            # the mapping but in their code they use the style image resized
            _, rh, rw, _ = net[args.pass2_reference_layer].shape
            style_img_resized = cv2.resize(style_img[0], (rw, rh))
            mapping_layers = set(args.pass2_style_layers + [args.pass2_reference_layer])
            style_activation_layers_pass2, duplicate_mask = \
                rearrange_style_activations_pass2(dict([(l, style_activation_layers[l]) for l in mapping_layers]),
                                                  mapping_pass1[args.pass2_reference_layer],
                                                  args.pass2_reference_layer,
                                                  tight_mask,
                                                  style_img_resized)

            loss_c = content_loss_pass2(net, mask, content_activation_layers)
            loss_s = style_loss_pass2(net, mask, duplicate_mask, style_activation_layers_pass2)
            loss_tv = tf.image.total_variation(net['input'] * mask['input'])
            loss_hist = tf.constant(0.0)  # histogram_loss(net, mask, style_activation_layers_pass2, duplicate_mask)
            loss = args.pass2_content_weight*loss_c + args.pass2_style_weight*loss_s + args.pass2_tv_weight*loss_tv \
                + args.pass2_hist_weight*loss_hist

            sess.run(tf.global_variables_initializer())
            sess.run(net['input'].assign(content_img))  # We begin with the unmodified content img (not the pass 1 out)

            print("Loading optimizer...")

            # We use the LBFGS optimizer. To get info about progress, we need to define a callback function
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',
                                                               options={'maxiter': args.iterations_pass2,
                                                                        'eps': args.pass2_learning_rate})

            def step_callback(_):
                global _iter
                if args.print_loss <= 0:
                    # Show progress bar
                    if _iter == 0:
                        print("")
                    percentage = (_iter + 1) / float(args.iterations_pass2)
                    filled_length = int(round(50 * percentage))
                    bar = '█' * filled_length + '-' * (50 - filled_length)
                    sys.stdout.write("\rPass 2: |{}| {:0.1f}%".format(bar, percentage * 100))
                    sys.stdout.flush()
                    if _iter == args.iterations_pass2-1:
                        print("\n")
                _iter += 1

            def loss_callback(l_c, l_s, l_tv, l_hist):
                global _iter
                if args.print_loss > 0 and _iter % args.print_loss == 0:
                    print("{}: content={}, style={}, tv={}, hist={}".format(_iter, l_c, l_s, l_tv, l_hist))

            _iter = 0
            optimizer.minimize(sess, step_callback=step_callback, loss_callback=loss_callback,
                               fetches=[loss_c, loss_s, loss_tv, loss_hist])
            generated_image = sess.run(net['input'])

            # Apply mask again to reconstruct image
            output_mask = cv2.GaussianBlur(mask_img, (3, 3), 1)  # Smooth edges
            output_mask = np.stack((output_mask,) * 3, axis=-1)  # To RGB
            output_mask = np.expand_dims(output_mask, 0)  # Add batch dimension

            output_pass2 = generated_image * output_mask + style_img * (1 - output_mask)
            # Save image
            file, extension = os.path.splitext(args.output_path)
            cv2.imwrite(file + "_pass2" + extension, postprocess_img(output_pass2))

        # Postprocessing
        if skip_post:
            print("Skipping post processing")
        else:
            post = run_postprocessing(output_pass2, content_img, style_img, num_cores=args.num_cores)

            # Apply mask again to reconstruct image
            output_mask = cv2.GaussianBlur(mask_img, (3, 3), 1)  # Smooth edges
            output_mask = np.stack((output_mask,) * 3, axis=-1)  # To RGB
            output_mask = np.expand_dims(output_mask, 0)  # Add batch dimension

            output_post = post * output_mask + style_img * (1 - output_mask)

    # Save final result
    if not skip_post:
        cv2.imwrite(args.output_path, postprocess_img(output_post))
    elif not skip_pass2:
        cv2.imwrite(args.output_path, postprocess_img(output_pass2))
    else:
        cv2.imwrite(args.output_path, postprocess_img(output_pass1))

    print("Done. Took {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    args = parser.parse_args()
    main()
