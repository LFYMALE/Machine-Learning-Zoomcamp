import pickle

from flask import Flask
from flask import request
from flask import jsonify

model = "pipeline.bin"

with open(model, "rb") as f:
    dv, model = pickle.load(f)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customers = request.get_json()

    x = dv.transform([customers])
    y_pred = model.predict_proba(x)[0,1]
    target = y_pred >= 0.5

    result = {
        'target_probability' : float(round(y_pred,3)),
        'target' : int(target)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9697)