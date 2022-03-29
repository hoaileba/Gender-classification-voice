import json, requests
from io import BytesIO
from flask import Flask, request, jsonify,url_for, Blueprint,render_template
import logging
from . import create_app,socketio
from flask_socketio import SocketIO, emit
app = create_app()
socketio = SocketIO(app,cors_allowed_origins="*")
# from .Api.Fake import Database
# database = Database()

import json

print("access")
@socketio.on('record', namespace = '/chat')
def Init(message):
    print("recive_message")
    sender = requests.get("http://localhost:5005/apis/init")
    x = sender.json()
    print("X:",x)
    emit('status', {'sender_id':x['sender'], 'text': x['text']})

