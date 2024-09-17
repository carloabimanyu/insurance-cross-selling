from flask import Flask, render_template, request, send_from_directory
import pickle

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

with open('model/insurance_cross_sell_model_v.0.1.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    data = {
        'Gender': '',
        'Age': '',
        'Driving_License': '',
        'Previously_Insured': '',
        'Vehicle_Age': '',
        'Vehicle_Damage': ''
    }
    prediction = None
    if request.method == 'POST':
        data = {
            'Gender': request.form['Gender'],
            'Age': request.form['Age'],
            'Driving_License': request.form['Driving_License'],
            'Previously_Insured': request.form['Previously_Insured'],
            'Vehicle_Age': request.form['Vehicle_Age'],
            'Vehicle_Damage': request.form['Vehicle_Damage']
        }

        input_data = [[
            int(data['Gender']), int(data['Age']), int(data['Driving_License']),
            int(data['Previously_Insured']), int(data['Vehicle_Age']), int(data['Vehicle_Damage'])
        ]]

        prediction = loaded_model.predict(input_data)
        prediction = "Customer is interested" if prediction[0] == 1 else "Customer is not interested"

    return render_template('index.html',data=data, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
