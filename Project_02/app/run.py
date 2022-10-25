import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
print('Loading data')
engine = create_engine('sqlite:///../data/ETL_Pipeline.db')
df = pd.read_sql_table('ETL_Pipeline', engine)

# load model
print('Loading classfier model')
model = joblib.load("../models/ML_Classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    flood_counts = df.groupby('floods').count()['message']
    flood_names = list(flood_counts.index)
    
    storm_counts = df.groupby('storm').count()['message']
    storm_names = list(storm_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        { # Visualization 1
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        { # Visualization 2
            'data': [
                Bar(
                    x=flood_names,
                    y=flood_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages categorized as Storm',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Floods"
                }
            }
        },
        { # Visualization 3
            'data': [
                Bar(
                    x=storm_names,
                    y=storm_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages categorized as Storm',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Storm"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
