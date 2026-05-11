"""Flask 应用入口"""
from flask import Flask
from flask_cors import CORS
from api.routes import api, init_inference
from models.inference import ModelInference


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    inference = ModelInference()
    init_inference(inference)
    app.register_blueprint(api)
    return app


if __name__ == '__main__':
    app = create_app()
    print("Starting Flask API on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
