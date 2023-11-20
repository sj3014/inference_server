from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
import tensorflow as tf
import logging
import json
import time
from preprocess import preprocess

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

loaded_model = tf.saved_model.load('./combined_slow')
print(loaded_model.signatures)

@app.route('/infer', methods=['POST'])
def index():
    body = request.get_json()
    frame_data = body
    input_data = preprocess(frame_data)

    # loaded_model = tf.saved_model.load('./combined_slow')

    
    s = time.time()
    predictions = loaded_model(input_data)
    e = time.time()

    print((e-s) * 1000)

    first = tf.reshape(predictions[-2], [-1])
    threeD = tf.reshape(predictions[-1], [-1])

    return {'first': first.numpy().tolist(), 'threeD': threeD.numpy().tolist()}

@socketio.on('connect')
def handle_connect():
    socketio.emit('message', 'hi')


@socketio.on('frame')
def handle_frame(frame_data):

    input_data = preprocess(frame_data)

    # loaded_model = tf.saved_model.load('./combined_slow')

    print('hi')
    
    s = time.time()
    predictions = loaded_model(input_data)
    e = time.time()

    first = tf.reshape(predictions[-2], [-1])
    threeD = tf.reshape(predictions[-1], [-1])

    socketio.emit('frame_res', {'first': first.numpy().tolist(), 'threeD': threeD.numpy().tolist()})

@socketio.on('hello')
def handle_array(data):
    global count
    print(count)

    frame_data = json.load(data)
    # image_data = base64.b64decode(frame_data.split(',')[1])  # Extract the base64 encoded part
    # image = Image.open(io.BytesIO(image_data))
    # image.save(f'/Users/jo/vifiveai/images/image{count}.png')
    # print(count)
    # image = image.resize((width, height))  # Resize as needed
    # image = tf.keras.preprocessing.image.img_to_array(image)
    # image_array = np.array(image)
    tensor = tf.convert_to_tensor(frame_data)
    # flipped_tensor = tf.image.flip_left_right(tensor)
     # Convert the flipped tensor back to a PIL Image
    flipped_image = tf.keras.preprocessing.image.array_to_img(tensor.numpy())
    count += 1

    # Save the flipped image
    flipped_image.save(f'/Users/jo/vifiveai/images/flipped_image{count}.png')
    # print(tensor)
    socketio.emit('array_res', 'frame')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000)
