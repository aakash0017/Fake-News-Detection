import base64
from io import BytesIO
from flask import Flask, redirect, url_for, request, jsonify, render_template
from flask.helpers import make_response
from flask.wrappers import Response
import boto3

from model import Sentence_Similarity

client = boto3.client('textract')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('recreation.html')


@app.route('/predict', methods=['POST'])
def process():
    req = request.get_json()
    
    recieved_bytes = bytes(req['image_base64'], 'utf-8')
    Context_sentence = req['Sentence_a']
    # Objective_sentence = req['Sentence_b']
    
    print(Context_sentence)
    
    # # create object 
    sim_checker = Sentence_Similarity(Context_sentence)
    ref = [
        "Former President Barack Obama to make his first appearance on The Late Late Show with James Corden",
        "Michelle Obama’s advice for coping with depression: ‘Develop your own tools’ and give yourself a break",
        "Barack and Michelle Obama Pay Tribute to Their Family Dog, Bo, Who Died This Weekend",
        "President Barack Obama To Make Debut Appearance On ‘The Late Late Show With James Corden'"
    ]
    sim_checker.getRefrences(list_references=ref)
    most_sim = sim_checker.getMostSimilar()
    print(most_sim)
    
    text = image_2_text(recieved_bytes)
    print(text)
    res = make_response(jsonify({"message": "JSON received"}), 200)
    return res
    
def image_2_text(receivedBytes_base64):
    binary_image = bytearray(base64.decodebytes(receivedBytes_base64))
    text = ''
    response = client.detect_document_text(Document={'Bytes': binary_image})
    
    # parse the respone 
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            text += item["Text"] + ' '
            
    return text
    
if __name__ == "__main__":
    app.run(debug=True)