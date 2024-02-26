from flask import Flask, render_template, request, jsonify

from process import search_and_recommend
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)


import glob
import joblib
import os
import requests

# Pattern to match TF-IDF matrix files
matrix_files = glob.glob('tfidf_matrix_large_*.z')
# Sort matrix files by modification time (most recent first)
matrix_files.sort(key=os.path.getmtime, reverse=True)
# Get the most recent matrix file
latest_matrix_file = matrix_files[0] if matrix_files else None

# Pattern to match TF-IDF vectorizer files
vectorizer_files = glob.glob('tfidf_vectorizer_large_*.joblib')
# Sort vectorizer files by modification time (most recent first)
vectorizer_files.sort(key=os.path.getmtime, reverse=True)
# Get the most recent vectorizer file
latest_vectorizer_file = vectorizer_files[0] if vectorizer_files else None

# Load the TF-IDF matrix
tfidf_matrix = joblib.load(latest_matrix_file) if latest_matrix_file else None

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load(latest_vectorizer_file) if latest_vectorizer_file else None

# Ensure both tfidf_matrix and tfidf_vectorizer are loaded
if tfidf_matrix is None or tfidf_vectorizer is None:
    raise Exception("Failed to load the TF-IDF matrix and/or vectorizer. Please check the files.")


@app.route('/')
def index():
    return render_template('frontend/index.html')


@app.route('/search', methods=['POST'])
def search():
    print("reach")
    import pandas as pd
    data = request.json
    keyword = data['query']
    search_type = data['type']
    
    final_df = pd.read_csv("final_df_large.csv")
    print(data)
    if search_type == 'user':
        results_df, _ = search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df)
        results = results_df.to_dict(orient='records')
    elif search_type == 'post':
        _, results_df = search_and_recommend(keyword, tfidf_vectorizer, tfidf_matrix, final_df)
        results = results_df.to_dict(orient='records')
    else:
        results = []
    print(results)

    

    for i in range(len(results)):
        if "DID" not in results[i]:
            continue
        try:
            url = f"https://bsky.social/xrpc/com.atproto.repo.listRecords?repo={results[i]['DID']}&collection=app.bsky.actor.profile&limit=100"

            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if "records" in data and len(data["records"]) > 0:
                    results[i]["actor"] = data["records"][0]["value"]
        except:
            continue
                

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
