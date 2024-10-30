from flask import Flask, redirect, render_template, request


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect("/")
    files = request.files.getlist("file")
    datasets = {}


if __name__ == "__main__":
    app.run(debug=True)
