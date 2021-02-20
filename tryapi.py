import base64
from io import BytesIO
import json
import re
from flask import Flask, redirect, url_for, request, jsonify, render_template
from flask.helpers import make_response
import boto3

client = boto3.client('textract')

app = Flask(__name__) 

@app.route('/')
def home():
   return render_template('recreation.html')

@app.route('/predict', methods=['POST'])
def predict():
   req = request.get_json()
   print(req['image_base64'])
   rec_bytes = bytes(req['image_base64'], 'utf-8')
   text = image_2_text(rec_bytes)
   print(text)
   res = make_response(jsonify({"messgae" : "JSON received"}), 200)
   return res

def image_2_text(receivedBytes_base64):
   # with open(documentName, 'rb') as document:
   #  imageBytes = bytearray(document.read())
   # convert base64 to binary (bytearray)
   binary_image = bytearray(base64.decodebytes(receivedBytes_base64))
   text = ''
   response = client.detect_document_text(Document={'Bytes': binary_image})
   for item in response["Blocks"]:
      if item["BlockType"] == "LINE":
         text += '\033[94m' +  item["Text"] + '\033[0m'
   return text

if __name__ == '__main__': 
   app.run(debug = True)