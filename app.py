"""
TRIBE Decoder — Flask Frontend + Proxy
Reads MODAL_ENDPOINT from environment (set in Railway or .env locally).
Optionally reads ACCESS_KEY — if set, all /analyze calls require the header
  X-Access-Key: <value>  or query param  ?key=<value>

Local dev:
    export MODAL_ENDPOINT=https://lgravina--tribe-decoder-web.modal.run
    python app.py
    Open: http://localhost:5001
"""

import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def get_modal_url() -> str:
    return os.environ.get("MODAL_ENDPOINT", "").rstrip("/")


def check_access() -> bool:
    """Return True if access is allowed. Always True when ACCESS_KEY is not set."""
    key = os.environ.get("ACCESS_KEY", "")
    if not key:
        return True
    provided = (
        request.headers.get("X-Access-Key", "")
        or request.args.get("key", "")
        or (request.get_json(silent=True) or {}).get("access_key", "")
    )
    return provided == key


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    modal_url = get_modal_url()
    if not modal_url:
        return jsonify({"local": "ok", "modal": "not_configured"})
    try:
        resp = requests.get(f"{modal_url}/health", timeout=60)
        resp.raise_for_status()
        return jsonify({"local": "ok", "modal": "connected", **resp.json()})
    except requests.exceptions.ConnectionError:
        return jsonify({"local": "ok", "modal": "unreachable"})
    except Exception as e:
        return jsonify({"local": "ok", "modal": "error", "error": str(e)})


@app.route("/analyze", methods=["POST"])
def analyze():
    if not check_access():
        return jsonify({"success": False, "error": "Unauthorized."}), 401

    modal_url = get_modal_url()
    if not modal_url:
        return jsonify({
            "success": False,
            "error": "MODAL_ENDPOINT not set. Add it as an environment variable."
        }), 400

    data = request.get_json(silent=True) or {}

    try:
        resp = requests.post(
            f"{modal_url}/analyze",
            json=data,
            timeout=600,
        )
        try:
            result = resp.json()
        except Exception:
            return jsonify({
                "success": False,
                "error": f"Modal returned an empty response (HTTP {resp.status_code}). "
                         "The request likely timed out — try shorter text or redeploy with modal deploy."
            }), 504
        return jsonify(result), resp.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Cannot reach Modal endpoint. Check that the deployment is live."
        }), 503

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Request timed out after 10 minutes. Try shorter text (< 500 words)."
        }), 504

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    modal_url = get_modal_url()
    access_key = os.environ.get("ACCESS_KEY", "")
    print()
    print("=" * 55)
    print("  TRIBE Decoder — Local Interface")
    print("=" * 55)
    print(f"  Open:       http://localhost:5001")
    print(f"  Backend:    {modal_url or '(MODAL_ENDPOINT not set)'}")
    print(f"  Access key: {'set' if access_key else 'not set (open access)'}")
    print("=" * 55)
    print()
    app.run(port=5001, debug=False)
