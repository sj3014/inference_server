import base64
import io
import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess(image_data):
    decoded_data = base64.b64decode(image_data.split(',')[1])  # Extract the base64 encoded part
    image = Image.open(io.BytesIO(decoded_data))
    image_array = np.array(image)
    
    tensor = tf.convert_to_tensor(image_array)

    expanded_tensor = tf.expand_dims(tensor, axis=0)

    boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
    resized_tensor = tf.image.crop_and_resize(expanded_tensor, boxes, box_indices=[0], crop_size=[384, 384])
    
    flipped_tensor = tf.image.flip_left_right(resized_tensor)

    squeezed_tensor = tf.squeeze(flipped_tensor, axis=0)  # Assuming you're removing the first dimension

    # Convert to integer type (if needed)
    squeezed_tensor = tf.cast(squeezed_tensor, tf.int32)

    # RGB to BGR conversion
    bgr_tensor = tf.reverse(squeezed_tensor, axis=[-1])

    # Expand the dimensions of the tensor
    expanded_tensor = tf.expand_dims(bgr_tensor, axis=0)

    # Convert to float
    input_data = tf.cast(expanded_tensor, tf.float32)

    return input_data