from flask import Flask, render_template
import keras
from keras import Model
from werkzeug import secure_filename
#from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from flask import request


# create the application object
app = Flask(__name__)

#inception = InceptionV3(weights='imagenet')
#model_new = Model(inception.input, inception.layers[-2].output)

#model = Model()
#model = keras.load_weights('model_30.h5')

def caption_predict(image_enc):
    caption = model.predict(image)
    return caption

def preprocess_and_predict(image):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    x -= 0.5
    x *= 2.
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    text = caption_predict(temp_enc)
    return text

# use decorators to link the function to a url
@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
      f = request.files['image']
      f.save(secure_filename(f.filename))
      print('file uploaded successfully')
#      capt = preprocess_and_predict(f)
    caps = ['Ashish', 'is', 'nice']
    return render_template('index.html', captions=caps)  # return a string

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
