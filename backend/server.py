import base64
import binascii
import pickle
import numpy as np
from keras.utils import pad_sequences, load_img, img_to_array
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.models import Model, load_model
from flask import Flask,render_template,request,jsonify, send_from_directory
from flask_cors import CORS

app=Flask(__name__,static_folder="build/static",template_folder="build")
CORS(app)

model = load_model('./model1.h5')
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
                #print(feature_vector.shape)
                return feature_vector
            processed_photo=encode_image(photo)
            print(processed_photo)
            processed_photo_new=processed_photo.reshape((1,2048))
            print(processed_photo_new.shape)
            in_text = "startseq"
            for i in range(35):
                sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
                sequence = pad_sequences([sequence],maxlen=35,padding='post')
                
                ypred = model.predict([processed_photo_new,sequence])
                ypred = ypred.argmax()
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
    app.run(debug=True)

