"""Flask app entry point. Serves the REST API and the static frontend."""
from __future__ import annotations

from flask import Flask, send_from_directory

from src.api.routes import api_bp
from src.config import PROJECT_ROOT, pipeline_cfg

FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.register_blueprint(api_bp)

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=pipeline_cfg.api_port, debug=False)
