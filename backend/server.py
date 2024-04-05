import base64
import pickle
import numpy as np
import json
import requests
from keras.utils import pad_sequences, load_img, img_to_array
from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS

app=Flask(__name__,static_folder="build/static",template_folder="build")
CORS(app)

modelres = ResNet50(weights="imagenet",input_shape=(224,224,3))
model_new = Model(modelres.input,modelres.layers[-2].output)

with open('saved_idx_to_wordy.pkl', 'rb') as f:
    idx_to_word = pickle.load(f)
with open('saved_word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

@app.route('/')
def serve():
    return render_template('index.html')

@app.route('/api', methods=['GET','POST'])
def process_data():
    try:
        data = request.get_json()
        image_data = data.split('base64,')[1]
        with open('uploaded_image.jpg', 'wb') as fh:
            fh.write(base64.decodebytes(bytes(image_data, "utf-8")))
        processed_data=0
        def predict_caption(photo):
            def preprocess_img(img):
                img = load_img(img,target_size=(224,224))
                img = img_to_array(img)
                img = np.expand_dims(img,axis=0)
                # Normalisation
                img = preprocess_input(img)
                return img
            
            def encode_image(img):
                img = preprocess_img(img)
                feature_vector = model_new.predict(img)
                feature_vector = feature_vector.reshape((-1,))
                return feature_vector
            
            processed_photo=encode_image(photo)
            processed_photo_new=processed_photo.reshape((1,2048))
            in_text = "startseq"
            for i in range(35):
                sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
                sequence = pad_sequences([sequence],maxlen=35,padding='post')
                tbposted={"instances":[{"input_2":processed_photo_new.tolist()[0],
                                        "input_3":sequence.tolist()[0]}]}
                res = requests.post('http://tfserving:8501/v1/models/model:predict', json=tbposted)
                ypred=json.loads(res.text)
                ypred=ypred["predictions"][0]
                ypred = np.array(ypred).argmax()
                word = idx_to_word[ypred]
                in_text += (' ' + word)
                if word == "endseq":
                    break

            final_caption = in_text.split()[1:-1]
            final_caption = ' '.join(final_caption)
            return final_caption
        processed_data=predict_caption("uploaded_image.jpg")
    except:
        processed_data = "Oops something went wrong 😔 Please reload the page and try again"
    return jsonify({'processed_data': processed_data})


if(__name__=="__main__"):
    app.run(host='0.0.0.0')

