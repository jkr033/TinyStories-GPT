import importlib.util, os, threading
import torch
from flask import Flask, request, jsonify, send_from_directory
from tokenizers import Tokenizer 

HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("tsg", os.path.join(HERE, "TinyStories-GPT.py"))
tsg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsg)

tok = Tokenizer.from_file(os.path.join(HERE, tsg.TOK))
ck = torch.load(os.path.join(HERE, tsg.CKPT), map_location=tsg.device)
model = tsg.GPT(**ck["cfg"]).to(tsg.device)
model.load_state_dict(ck["state"])
model.eval()
print("Model loaded.")

app = Flask(__name__)
lock = threading.Lock()

MAX_PROMPT_CHARS = 500

def number(value, default, low, high, cast):
    try:
        return min(max(cast(value), low), high)
    except (TypeError, ValueError):
        return default

@app.get("/")
def index():
    return send_from_directory(HERE, "chat-ui.html")     
 
 
@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}             
    prompt = data.get("prompt")


    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Write the start of the story first."), 400
    if len(prompt) > MAX_PROMPT_CHARS:
        return jsonify(error=f"Too long. Keep it under {MAX_PROMPT_CHARS} characters."), 400

    max_new = number(data.get("max_new"), 150, 1, 300, int)
    temp = number(data.get("temperature"), 0.8, 0.1, 1.5, float)

    with lock:
        story = tsg.complete(model, tok, prompt.strip(), max_new=max_new, temperature=temp)
    return jsonify(story=story)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)