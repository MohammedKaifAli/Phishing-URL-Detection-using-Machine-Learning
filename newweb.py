from flask import Flask, render_template, request, redirect, url_for
import os
import csv
import joblib
import numpy as np
import joblib
joblib.dump(model, "phishing_detection_model.pkl")


app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
def index():
    urls_data = read_last_10_urls()
    if request.method == 'POST':
        u_input = request.form['user_input']
        result = check_url(u_input)
        return redirect(url_for('result', status=result['status'], description=result['description'], url=result['url'], confidence=result['confidence']))
    return render_template('index.html', urls_data=urls_data)

@app.route('/result')
def result():
    result_data = {
        'status': request.args.get('status'),
        'description': request.args.get('description'),
        'url': request.args.get('url'),
        'confidence': request.args.get('confidence')
    }
    urls_data = read_last_10_urls()
    return render_template('result.html', result=result_data, urls_data=urls_data)

def check_url(input_url):
    clf = joblib.load('phishing_detection_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    url_features = vectorizer.transform([input_url])
    prediction = clf.predict(url_features)
    prediction_proba = clf.predict_proba(url_features)

    status = "Phishing" if prediction[0] == "bad" else "Legitimate"
    confidence = np.max(prediction_proba) * 100
    report = generate_report(input_url, status, confidence)

    output = {'url': input_url, 'description': report, 'status': status, 'confidence': f"{confidence:.2f}%"}
    save_to_csv([input_url, report, status, confidence])
    return output

def generate_report(url, status, confidence):
    if status == "Phishing":
        return f"The URL is detected as phishing with a confidence of {confidence:.2f}%. Common indicators include unusual subdomains, suspicious parameters, or known malicious patterns."
    else:
        return f"The URL is detected as legitimate with a confidence of {confidence:.2f}%."

def read_last_10_urls():
    file_path = 'data.csv'
    urls = []
    if not os.path.exists(file_path):
        return urls

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader, None)
        if header is None:
            return urls

        rows = list(csvreader)
        if len(rows) == 0:
            return urls

        for row in reversed(rows[-10:]):
            if len(row) < 4:
                # Handle rows with missing columns
                url, description, status, confidence = (row + [''] * 4)[:4]
            else:
                url, description, status, confidence = row
            urls.append({'url': url, 'description': description, 'status': status, 'confidence': confidence})
    return urls

def save_to_csv(data):
    file_path = 'data.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['URL', 'Description', 'Status', 'Confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'URL': data[0], 'Description': data[1], 'Status': data[2], 'Confidence': data[3]})

if __name__ == '__main__':
    app.run(debug=True)