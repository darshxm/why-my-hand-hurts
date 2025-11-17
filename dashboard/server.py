import argparse
import getpass
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from cryptography.fernet import Fernet, InvalidToken

# Ensure project root is on sys.path so we can import analytics when run from dashboard/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics import get_encryption_key, load_data


def _load_fernet(password: str, salt_path: str) -> Fernet:
    if not os.path.exists(salt_path):
        raise FileNotFoundError(f"Salt file not found at {salt_path}")
    with open(salt_path, "rb") as fh:
        salt = fh.read()
    key = get_encryption_key(password, salt)
    return Fernet(key)


def _build_app(keylog_path: str, fernet: Fernet) -> Flask:
    static_dir = ROOT / "dashboard"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="")

    @app.after_request
    def add_cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    def _filtered_df():
        return load_data(
            keylog_path,
            fernet,
            app_filter=request.args.get("app"),
            window_filter=request.args.get("window"),
        )

    @app.get("/api/summary")
    def summary():
        df = _filtered_df()
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df["app"] = df["app"].fillna("").replace("", "<unknown>")
        df["window_title"] = df["window_title"].fillna("").replace("", "<unknown>")

        app_counts = (
            df.groupby("app").size().sort_values(ascending=False).reset_index(name="count")
        )
        window_counts = (
            df.groupby("window_title")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="count")
        )
        app_durations = (
            df.groupby("app")["duration"].mean().sort_values(ascending=False).reset_index()
        )

        return jsonify(
            {
                "topApps": app_counts.to_dict(orient="records"),
                "topWindows": window_counts.to_dict(orient="records"),
                "avgDurations": app_durations.to_dict(orient="records"),
                "totalKeystrokes": int(len(df)),
            }
        )

    @app.get("/api/key-frequency")
    def key_frequency():
        df = _filtered_df()
        counts = df["key"].value_counts().reset_index()
        counts.columns = ["key", "count"]
        return jsonify({"keys": counts.to_dict(orient="records")})

    @app.get("/api/key-durations")
    def key_durations():
        df = _filtered_df()
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        durations = (
            df.groupby("key")["duration"].mean().sort_values(ascending=False).reset_index()
        )
        durations.columns = ["key", "avgDuration"]
        return jsonify({"keys": durations.to_dict(orient="records")})

    @app.get("/api/metadata")
    def metadata():
        info = {}
        if os.path.exists(keylog_path):
            stat = os.stat(keylog_path)
            info = {
                "keylogPath": os.path.abspath(keylog_path),
                "lastModified": stat.st_mtime,
                "sizeBytes": stat.st_size,
            }
        return jsonify(info)

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve keylog analytics over HTTP.")
    parser.add_argument("keylog_file", nargs="?", default="keylog.csv", help="Path to keylog CSV")
    parser.add_argument("--salt", default="key.salt", help="Path to salt file")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    args = parser.parse_args()

    password = os.environ.get("KEYLOG_PASSWORD") or getpass.getpass(
        "Enter the password to decrypt the keylog: "
        )

    try:
        fernet = _load_fernet(password, args.salt)
    except InvalidToken:
        raise SystemExit("Invalid password for decrypting keylog.")
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    app = _build_app(args.keylog_file, fernet)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
