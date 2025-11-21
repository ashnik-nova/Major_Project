from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

from routes.uploads_routes import upload_bp
from routes.dogcat_routes import dogcat_bp
from routes.tb_routes import tb_bp
# from routes.explain_routes import explain_bp 
from routes.explain_shap_routes import gradcam_shap_bp


def create_app():
    app = Flask(__name__)

    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(dogcat_bp, url_prefix="/api/predict/dogcat")
    app.register_blueprint(tb_bp, url_prefix="/api/predict/tb")
    # app.register_blueprint(explain_bp, url_prefix="/api")
    app.register_blueprint(gradcam_shap_bp,url_prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
