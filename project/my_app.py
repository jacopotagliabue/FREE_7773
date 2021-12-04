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
# to make a prediction on an unseen data point
vectorizer = pickle.load(open('vectorizer.pkl','rb+'))
model = pickle.load(open('model.pkl','rb+'))

@app.route('/',methods=['POST','GET'])
def main():

  # on GET we display the page  
  if request.method=='GET':
    return render_template('index.html')
  # on POST we make a prediction over the input text supplied by the user
  if request.method=='POST':
    # Converting all the form values to float and making them append in a list(features)
    features=[float(x) for x in request.form.values()]
    # Debug: check we have the right input
    print(features)
    # Predicting the label for the features collected
    labels=model.predict([features])
    # Printing the labels array for debug purpose
    print(labels)
    # Storing the result from the labels array
    species=labels[0]
    # Returning the response to ajax	
    return s
    
if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)