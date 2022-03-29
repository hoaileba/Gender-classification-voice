import json
from flask import request, jsonify, Blueprint,render_template
from flask import Flask,current_app
import re
import requests
from flask.views import MethodView
from flask_restful import Resource, Api
from flask import Blueprint
from MyProj.Api import apiBp
from random import randint


# database = Database()
api = Api(apiBp)

class Init(Resource):
    def get(self):
    
       
        return jsonify({
            'sender': "1",
            'text' :"abc"
        })

api.add_resource(Init,'/apis/init')

