
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# 加载模型
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 加载特征处理函数
from feature import seq_to_features_with_properties

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        sequence = request.form["sequence"]
        features = np.array([seq_to_features_with_properties(sequence, 200)])
        prediction = model.predict(features)[0]
        result = "Umami" if prediction == 1 else "Not Umami"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
