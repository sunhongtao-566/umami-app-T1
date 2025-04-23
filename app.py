from flask import Flask, request, render_template
import pickle
import numpy as np
from feature import seq_to_features_with_properties

app = Flask(__name__)

import joblib

# 使用 joblib 加载模型和选择器
model = joblib.load("model.pkl")
selector = joblib.load("selector.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        sequence = request.form["sequence"]
        max_len = 200
        features = np.array([seq_to_features_with_properties(sequence, max_len)])
        reduced = selector.transform(features)
        result = model.predict(reduced)
        prediction = "Umami" if result[0] == 1 else "Not Umami"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
