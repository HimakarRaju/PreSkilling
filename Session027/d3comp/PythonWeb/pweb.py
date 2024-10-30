from flask import Flask, render_template, request, jsonify
import os
import pandas as pd

app = Flask(__name__)

DATASET_FOLDER = "datasets"


@app.route("/")
def index():
    datasets = os.listdir(DATASET_FOLDER)
    return render_template("index.html", datasets=datasets)


@app.route("/analyze", methods=["POST"])
def analyze():
    file_name = request.form["file_name"]
    file_path = os.path.join(DATASET_FOLDER, file_name)

    data = pd.read_csv(file_path)

    columns = data.columns.tolist()
    data_summary = {
        "columns": columns,
        "types": data.dtypes.astype(str).tolist(),
        "sample_data": data.head(10).to_dict(orient="records"),
    }

    return jsonify(data_summary)


if __name__ == "__main__":
    app.run(debug=True)
