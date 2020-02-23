from flask import Flask, render_template, request

from dialogue_manager import DialogueManager
from utils import RESOURCE_PATH

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

paths = RESOURCE_PATH
dialogue = DialogueManager(paths)

@app.route("/get")
def get_bot_response():
    user_question = request.args.get('msg')
    return dialogue.generate_answer(user_question)


if __name__ == "__main__":
    app.run()
