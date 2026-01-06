
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

def get_deforestation_outline(model, img_array, class_idx, threshold=0.2):
    """
    Returns an image with red outlines where deforestation is detected.
    """
    # Find last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    img_tensor = np.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = np.maximum(cam, 0)
    cam /= np.max(cam) + 1e-8

    # Resize CAM to original image
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize(
        (img_array.shape[1], img_array.shape[0])
    )
    cam_resized = np.array(cam_resized)

    # Threshold to get binary mask
    mask = cam_resized > int(threshold * 255)

    # Draw contours
    original_img = Image.fromarray(np.uint8(img_array * 255)).convert("RGB")
    draw = ImageDraw.Draw(original_img)

    # Draw red points where deforestation is detected
    ys, xs = np.where(mask)
    for (x, y) in zip(xs, ys):
        draw.point((x, y), fill=(255, 255, 255))

    return original_img