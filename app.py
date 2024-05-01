import os
import glob
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"

# Load model
model = tf.keras.models.load_model(STATIC_FOLDER + "/model.h5")

IMAGE_SIZE = 64

# Define the classes
classes = {
    0: 'Actinic keratoses and intraepithelial carcinomae',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Pyogenic granulomas and hemorrhage',
    6: 'Melanoma'
}

def load_and_preprocess_image():
    test_fldr = 'uploads'
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_fldr,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=1,
            class_mode=None,
            shuffle=False)
    test_generator.reset()
    return test_generator


# Predict & classify image
def classify(model):
    test_generator = load_and_preprocess_image()
    probs = model.predict_generator(test_generator, steps=len(test_generator))
    predicted_class = classes[tf.argmax(probs[0]).numpy()]
    classified_prob = tf.reduce_max(probs[0]).numpy()
    return predicted_class, classified_prob


# home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/all_class/*.*")
    for filePath in filelist:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)

        label, prob = classify(model)
        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()
