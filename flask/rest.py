from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import numpy as np
import pickle
import cv2
import flask
import magic
import gc

# import for extract color
from sklearn.cluster import KMeans
import utils
import collections

app = flask.Flask(__name__)

model_c = "./model_category/multiCategory.model"
labelbin_c = "./model_category/mlb.pickle"
model_t = "./model_texture/texture.model"
labelbin_t = "./model_texture/lb.pickle"
model_f = "./model_fabric/fabric.model"
labelbin_f = "./model_fabric/lb.pickle"

# modelc = load_model(model_c)
# lbc = pickle.loads(open(labelbin_c, "rb").read())
# modelt = load_model(model_t)
# lbt = pickle.loads(open(labelbin_t, "rb").read())
# modelf = load_model(model_f)
# lbf = pickle.loads(open(labelbin_f, "rb").read())

clt = KMeans(n_clusters = 2)    # 일단, 배경과 옷으로 생각


def pre_process_image(image, target):
    try:
        image = cv2.resize(image, target)
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    except:
        return False
    return image

def predict_multi_label(image, model, lb):
    try:
        proba = model.predict(image)[0]
        idx = np.argmax(proba)[::-1][:2]
        predicts = []
        case = 0
        for (i, j) in enumerate(idx):
            predicts[case] = lb.classes_[j]
            case += 1
            # label = "{}: {:.2f}%".format(lb.classes_[j], proba[j] * 100)
            # print("multi-label: "+label)
        label1 = predicts[0]
        label2 = predicts[1]
    except:
        return (False, 0)
    return (label1, label2)

def predict_single_label(image, model, lb):
    try:
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        # labelc = "{}: {:.2f}% ({})".format(label, probac[idx] * 100, "")
        # print("multi-label: "+label)
    except:
        return False
    return label

def predict_color(image):
    # extract color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt.fit(image)
    hist = utils.centroid_histogram(clt)

    d = {}
    for (percent, color) in zip(hist, clt.cluster_centers_):
        p = round(percent, 2)
        colors = [int(color[0]), int(color[1]), int(color[2])] # R: color[0], G: color[1], B: color[2]
        d[p] = colors

    od = collections.OrderedDict(sorted(d.items(), reverse=True))
    # print(od)
    R = 0
    G = 0
    B = 0
    count = 1 # 주석?
    for percent in od:
        if count > 2: break # 주석?
        color = od[percent]
        # suppose pure white or pure black is background
        if (color[0] < 5 and color[1] < 5 and color[2] < 5) or (color[0] > 250 and color[1] > 250 and color[2] > 250):
            # print("background")
            continue
        # print(str(count) + ": " + "R ("+str(color[0])+"), G ("+str(color[1])+"), B ("+str(color[2])+")")
        else:
            R = color[0]
            G = color[1]
            B = color[2]
            break
        count+=1 # 주석?
    return (R,G,B)

@app.route("/predict", methods=["POST"])
def predict():
    # time1 = tracemalloc.take_snapshot()
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            postImage = flask.request.files["image"].read()
            extention = magic.from_buffer(postImage).split()[0].upper()
            image = cv2.imread(postImage)
            color_image = image.copy()
            image = pre_process_image(image, target=(96, 96))
            if image is False:
                return flask.jsonify(data)

            # predict category, sub_category
            label_category1, label_category2 = predict_multi_label(image, modelc, lbc) 
            # predict texture
            label_texture = predict_single_label(image, modelt, lbt)
            # predict fabric
            label_fabric = predict_single_label(image, modelf, lbf)
            # predict color
            R, G, B  = predict_color(color_image)

            if label_category1 is False:
                label_category1 = "unknown"
                label_category2 = "unknown"
            if label_texture is False:
                label_texture = "unknown"
            if label_fabric is False:
                label_fabric = "unknown"

            clear_session()
            image = None
            postImage = None
            data["category1"] = label_category1
            data["category2"] = label_category2
            data["texture"] = label_texture
            data["fabric"] = label_fabric
            data["colorR"] = R
            data["colorG"] = G
            data["colorB"] = B
            data["success"] = True
            app.logger.info(data)
    gc.collect()
    return flask.jsonify(data)

# 연결 확인용
@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='114.70.23.80', port=1205)
