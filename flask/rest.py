from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
import tensorflow as tf
import numpy as np
import pickle
import cv2
import flask
import magic
import gc
import io
from base64 import encodebytes
from PIL import Image
import requests  # for remove bg
# import for extract color
from sklearn.cluster import KMeans
import utils
import collections

app = flask.Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# model_c = "./model_category/multiCategory.model"
# labelbin_c = "./model_category/mlb.pickle"
model_c = "./model_category/category.model"
labelbin_c = "./model_category/lb.pickle"
model_t = "./model_texture/texture.model"
labelbin_t = "./model_texture/lb.pickle"
model_f = "./model_fabric/fabric.model"
labelbin_f = "./model_fabric/lb.pickle"

modelc = load_model(model_c)
lbc = pickle.loads(open(labelbin_c, "rb").read())
modelt = load_model(model_t)
lbt = pickle.loads(open(labelbin_t, "rb").read())
# modelf = load_model(model_f)
# lbf = pickle.loads(open(labelbin_f, "rb").read())

clt = KMeans(n_clusters = 2)    # 일단, 배경과 옷으로 생각

REMOVEBG_API_KEY = "FgGa18D74Hk7Fe5UAt31bBnT"


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
        # print("label: ", label)
        # labelc = "{}: {:.2f}% ({})".format(label, probac[idx] * 100, "")
        # print("multi-label: "+label)
        return label
    except:
        return False


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
    return R,G,B

def getCategory(subCategory):
    top = ["blouse", "longTshirt", "shortTshirt", "sleeveless"]
    bottom = ["longPants", "shortPants", "skirt"]
    outer = ["cardigan&vest", "coat", "jacket", "jumper"]

    if subCategory in top:
        category = "top"
    elif subCategory in bottom:
        category = "bottom"
    elif subCategory in outer:
        category = "outer"
    else:
        category = subCategory

    return category


root_path_before = "upload/"
root_path_after = "result/"

def removeBackground(file_name, image):
    # 1) remove-bg 적용하지 않을 시: 1) 아래 주석 풀고, 2) 아래 주석처리
    # input_path = root_path_before + file_name + ".jpg"
    # cv2.imwrite(input_path, image)
    # return True, image
    # 2) remove-bg 적용 부분
    input_path = root_path_before + file_name + ".jpg"
    output_path = root_path_after + file_name + ".png"
    cv2.imwrite(input_path, image)
    # remove-bg
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(input_path, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': REMOVEBG_API_KEY},
    )
    if response.status_code == requests.codes.ok:
        with open(output_path, 'wb') as out:
            out.write(response.content)
            return True, cv2.imread(output_path)
    else:
        print("Error:", response.status_code, response.text)
        return False, image

def get_response_image(isSuccess, file_name):
    input_path = root_path_before + file_name + ".jpg"
    output_path = root_path_after + file_name + ".png"
    if isSuccess:
        image_path = output_path
    else:
        image_path = input_path

    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    label_category = "unknown"
    label_texture = "unknown"
    label_fabric = "unknown"
    R = 0
    G = 0
    B = 0
    isSuccess = False
    file_name = "temp" # TODO 이미지 이름 -> timestamp로 auto 생성
    if flask.request.files.get("image"):
        file = flask.request.files["image"]
        npimg = np.fromfile(file, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        isSuccess, image = removeBackground(file_name, image)
        color_image = image.copy()
        image = pre_process_image(image, target=(96, 96))
        if image is False:
            return flask.jsonify(data)

        # predict category, texture, (fabric), color
        label_category = predict_single_label(image, modelc, lbc)
        label_texture = predict_single_label(image, modelt, lbt)
        # label_fabric = predict_single_label(image, modelf, lbf)
        R, G, B = predict_color(color_image)
        clear_session()
        image = None
        data["success"] = True

    data["category"] = getCategory(label_category)
    data["subCategory"] = label_category
    data["texture"] = label_texture
    # data["fabric"] = label_fabric
    data["R"] = R
    data["G"] = G
    data["B"] = B

    encoded_img = get_response_image(isSuccess, file_name)
    data["encodedImage"] = encoded_img

    app.logger.info(data)
    gc.collect()
    return flask.json.dumps(data)

# 연결 확인용
@app.route('/')
def hello_world():
    data = {"error": False, "message": "Hello World"}
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='114.70.23.158', port=1205)
