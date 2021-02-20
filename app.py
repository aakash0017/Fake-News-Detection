from botocore.retries import base
from io import BytesIO
from flask import Flask, redirect, url_for, request, jsonify, render_template
from flask.helpers import make_response
from WMD import WordS_Mover_Distance
import boto3
import base64
client = boto3.client('textract')
app = Flask(__name__) 

@app.route('/')
def home():
   return render_template('recreation.html')

@app.route('/success', methods=['GET']) 
def success(): 
   dist = request.cookies.get('dist')
   return jsonify({"key" : dist}) 
   
@app.route('/predict', methods=['POST'])
def predict():
   # declare model object
   global wmd_Obj
   req = request.get_json()
   print(req)
   # convert requested url into base64 string
   text = image_2_text(req)
   # get json data using post method
   # s1 = request.json['sntnce_One']
   # s2 = request.json['sntnce_Two']
   # calculate distance b/w sentences 
   # dist = wmd_Obj.TextSimilarity(s1, s2)
   dist = text
   print(dist)
   res = make_response(jsonify({"messgae" : "JSON received"}), 200)
   # output
   # output = {"distance" : dist}
   # return data
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
   wmd_Obj = WordS_Mover_Distance()
   app.run(debug = True)