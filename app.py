#importing required libraries

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
from time import sleep

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)
CORS(app)


@app.route("/test/url", methods=["POST", "GET"])
def test_url():
    try:
        data = request.get_json()
        url = data["url"]
        print(url)
        # sleep(1)
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred = gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        response = {}
        result = {}
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        if round(y_pro_non_phishing, 2) >= 0.50:
            result["prediction"] = "safe"
            result["score"] = round(y_pro_non_phishing, 3) * 100
        else:
            result["prediction"] = "unsafe"
            result["score"] = 100 - (round(y_pro_non_phishing, 3) * 100)

        response["status"] = "success"
        response["message"] = result
        return jsonify(response)
    except Exception as e:
        response["status"] = "error"
        response["message"] = str(e)
        print("sent")
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
