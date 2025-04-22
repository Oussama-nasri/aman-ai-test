from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint
from .routes import api_bp
from flask_cors import CORS

def create_app():
    app = Flask(__name__, static_folder="../swagger")

    CORS(app)

    # Swagger UI served at /docs
    SWAGGER_URL = "/docs"
    API_URL = "/swagger/swagger.yml"  # this assumes swagger.yml is in /static
    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={'app_name': "SMILES Prediction API"}
    )
    app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

    # Register API routes under /api
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
