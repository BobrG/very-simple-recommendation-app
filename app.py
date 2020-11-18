from flask import Flask, render_template,flash, redirect,url_for,session,logging,request
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import KFold
from flask_sqlalchemy import SQLAlchemy
from collections import defaultdict
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database.db'
db = SQLAlchemy(app)
meta = pd.read_csv('/code/movies_metadata.csv')
meta.drop([19730, 29503, 35587], inplace=True)
meta['id'] = meta['id'].astype('int')
RECOMMENDATIONS = []


app.secret_key = "super secret key"

movie_title_db = np.array(meta.title)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(90))
    rating = db.Column(db.Integer)

@app.route('/', methods=['GET', 'POST'])
def main():
    global RECOMMENDATIONS

    if request.method == 'POST':
        if request.form.get('remove_button'):
            delete_choice = request.form['movie_to_delete']
            qry = db.session.query(User).filter(User.id == delete_choice)
            user = qry.first()
            if user:
                db.session.delete(user)
                db.session.commit()
        elif request.form.get('recommend_button'):
            recommendations = get_recommendations()
            RECOMMENDATIONS = recommendations
        elif request.form.get('add_button'):

            add_choice = request.form['movie_to_add']
            new_rating_choice = request.form['new_rate']
            if db.session.query(User.id).filter_by(name=add_choice).scalar() is not None:
                return render_template('main.html', movies=get_movies(), rating=[1, 2, 3, 4, 5], recommendations=RECOMMENDATIONS)

            db.session.add(User(name=add_choice, rating=new_rating_choice))
            db.session.commit()
            return render_template('main.html', movies=get_movies(), rating=[1, 2, 3, 4, 5], recommendations=RECOMMENDATIONS)
        else:
            movie_name = request.form['movie_name']
            rating_choice = request.form['ratings']

            if db.session.query(User.id).filter_by(name=movie_name).scalar() is not None:
                return render_template('main.html', movies=get_movies(), rating=[1, 2, 3, 4, 5], recommendations=RECOMMENDATIONS)

            if request.form.get('submit_button'):

                if movie_name in movie_title_db:
                    db.session.add(User(name=movie_name, rating=rating_choice))
                    db.session.commit()
                else:
                    flash("Movie not in the database :c")

        return render_template('main.html', movies=get_movies(), rating=[1, 2, 3, 4, 5], recommendations=RECOMMENDATIONS)

    return render_template('main.html', movies=get_movies(), rating=[1, 2, 3, 4, 5], recommendations=RECOMMENDATIONS)

def get_top_n(predictions, n=5):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_recommendations():
    id_map = pd.read_csv('/code/links_small.csv')[['movieId', 'tmdbId']].dropna()
    id_map['tmdbId'] = id_map['tmdbId'].apply(int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(meta[meta['id'].isin(id_map['id'])][['title', 'id']], on='id', how='inner').set_index('title')
    id_map_back = id_map.merge(meta[meta['id'].isin(id_map['id'])][['title', 'id']], on='id', how='inner').set_index('movieId')

    users_hist = pd.read_csv('/code/ratings_small.csv')
    users_hist.drop('timestamp', inplace=True, axis=1)
    query = User.query.all()
    userid = len(np.unique(users_hist.userId)) + 1
    users_films = []
    for u in query:
        try:
            movieid = id_map.loc[u.name].movieId
        except KeyError:
            continue
        if type(movieid) == pd.core.series.Series:
            tmp = pd.DataFrame([[userid, movieid[0], u.rating]], columns=['userId', 'movieId', 'rating'])
            users_films.append(movieid[0])
        else:
            tmp = pd.DataFrame([[userid, movieid, u.rating]], columns=['userId', 'movieId', 'rating'])
            users_films.append(movieid)
        users_hist = users_hist.append(tmp, ignore_index=True)
    reader = Reader()
    data = Dataset.load_from_df(users_hist, reader)
    algo = SVD(random_state=42)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    predictions = []

    for i in np.unique(id_map['movieId']):
        if i not in users_films:
            predictions.append(algo.predict(userid, i))

    recommendations = [id_map_back.loc[i].title for i, j in get_top_n(predictions)[userid]]
    return recommendations

def get_movies():
    query = User.query.all()
    return query

if __name__ == "__main__":
    db.create_all()
    app.run(host="0.0.0.0", debug=True)
