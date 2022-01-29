from flask import Flask, request, jsonify
import pandas as pd
import pickle
# import pmdarima as pm

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def predict():
    json_ = request.json
    # query_df = pd.DataFrame(json_)
    # query = pd.get_dummies(query_df)

    infile = open('datathon.pkl', 'rb')
    test_model = pickle.load(infile)

    infile.close()

    forecast = test_model.predict(10)
    print(forecast)
    
    # classifier = joblib.load('datathon.pkl')
    # prediction = classifier.predict(query)
    return jsonify({'prediction': forecast})


if __name__ == '__main__':
     app.run(port=8080)