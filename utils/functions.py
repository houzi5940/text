from App.app_views import main
from flask import Flask
import os

def create_app():
    BASE_DIR=os.path.dirname(os.path.dirname(__file__))
    static_dir=os.path.join(BASE_DIR,'static')
    app=Flask(__name__,static_folder=static_dir)
    app.register_blueprint(blueprint=main,url_prefix='/app')
    return app
