"""
    This script runs a small Flask app that displays a simple web form for users to insert a sentence they 
    want to classify with the model.

    Inspired by: https://medium.com/shapeai/deploying-flask-application-with-ml-models-on-aws-ec2-instance-3b9a1cec5e13
"""

from flask import Flask, render_template, request
import pickle
import numpy as np


# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')
# We need to load the pickled model file AND the vectorizer to transform the text 
# to make a prediction on an unseen data point - note that the script assumes the pickled files are in
# the samee folder
vectorizer = pickle.load(open('vectorizer.pkl','rb+'))
model = pickle.load(open('model.pkl','rb+'))

@app.route('/',methods=['POST','GET'])
def main():

  # on GET we display the page  
  if request.method=='GET':
    return render_template('index.html')
  # on POST we make a prediction over the input text supplied by the user
  if request.method=='POST':
    # debug
    # print(request.form.keys())
    input_sentence = request.form['sl']
    # make sure we lower case it
    final_sentence = input_sentence.lower()
    # debug
    # print(final_sentence)
    vectorized_sentence = vectorizer.transform([final_sentence])
    labels = model.predict(vectorized_sentence)
    #  debug
    print(labels)
    # Returning the response to ajax	
    return "Predicted label is {}".format(labels[0])
    
if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)