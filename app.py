"""
TRIBE Decoder — Local Proxy Server
Run: python app.py
Then open: http://localhost:5001
"""

import json
import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CONFIG_FILE = ".tribe_config.json"


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"ngrok_url": ""}


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)


@app.route("/")
def index():
    cfg = load_config()
    return render_template("index.html", ngrok_url=cfg.get("ngrok_url", ""))


@app.route("/config", methods=["POST"])
def update_config():
    data = request.get_json(silent=True) or {}
    cfg = load_config()
    cfg["ngrok_url"] = data.get("ngrok_url", "").rstrip("/")
    save_config(cfg)
    return jsonify({"success": True})


@app.route("/health", methods=["GET"])
def health():
    cfg = load_config()
    ngrok_url = cfg.get("ngrok_url", "").rstrip("/")
    if not ngrok_url:
        return jsonify({"local": "ok", "colab": "not_configured"})
    try:
        resp = requests.get(f"{ngrok_url}/health", timeout=10)
        resp.raise_for_status()
        return jsonify({"local": "ok", "colab": "connected", **resp.json()})
    except requests.exceptions.ConnectionError:
        return jsonify({"local": "ok", "colab": "unreachable"})
    except Exception as e:
        return jsonify({"local": "ok", "colab": "error", "error": str(e)})


@app.route("/analyze", methods=["POST"])
def analyze():
    cfg = load_config()
    ngrok_url = cfg.get("ngrok_url", "").rstrip("/")

    if not ngrok_url:
        return jsonify({
            "success": False,
            "error": "No Colab URL configured. Enter your ngrok URL in the config panel."
        }), 400

    data = request.get_json(silent=True) or {}

    try:
        resp = requests.post(
            f"{ngrok_url}/analyze",
            json=data,
            timeout=600,  # TRIBE inference can take several minutes on T4
        )
        return jsonify(resp.json()), resp.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Cannot reach Colab. Is Cell 7 running? Is the ngrok URL current?"
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Request timed out after 10 minutes. Try shorter text (< 500 words)."
        }), 504

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  TRIBE Decoder — Local Interface")
    print("=" * 50)
    print("  Open: http://localhost:5001")
    print("=" * 50)
    print()
    app.run(port=5001, debug=False)
