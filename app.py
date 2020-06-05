from __future__ import division,print_function
import sys
import os
import flask
from flask import Flask,redirect,url_for,render_template,request
from flask import Flask, Response, render_template, request

from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
from ast import literal_eval
import random
from urllib.request import urlopen
import re


app=Flask(__name__)
def clean_data(text):


    if isinstance(text, list):
      return [str.lower(i.replace(" ", "")) for i in text]

      
    else:
        #Check if director exists. If not, return empty string
        if isinstance(text, str):
          text=str.lower(text)
          text=re.sub("-","",text)
          text=re.sub(":","",text)
          text=re.sub("'","",text)
          text=re.sub(" ","",text)
          return text
        else:
            return ''
def prepare_data():
  movie=pd.read_csv("tmdb_5000_movies.csv")
  credit=pd.read_csv('tmdb_5000_credits.csv')
  data=pd.merge(movie,credit)
  #loading posters dataset
  poster=pd.read_csv("MovieGenre.csv",encoding="ISO-8859-1")
  #Remove years from the poster table like 'antman(2008)'
  title=[]
  for i in range(len(poster)):
    title.append(poster['Title'][i].split('(')[0])
  poster['Title']=title
  #cleaning titles from both the csv
  poster['Title1']=poster['Title'].apply(clean_data)
  data['Title1']=data['original_title'].apply(clean_data)
  #merging into final dataset
  data1=pd.merge(data,poster,left_on="Title1",right_on="Title1",how="inner")
  data1.drop_duplicates( subset=['Title1'] ,keep="first", inplace=True)
  data1.dropna(subset=["Poster","overview"],inplace=True)
  data1.drop(['homepage','tagline'],axis=1,inplace=True)
  # data1.to_csv('dataset.csv')
  return data1



def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

def Demographic_filtering(data1):
  C= data1['vote_average'].mean()
  m= data1['vote_count'].quantile(0.9)

  q_movies = data1.copy().loc[data1['vote_count'] >= m]
  def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
  q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

  q_movies = q_movies.sort_values('score', ascending=False)

  #Print the top 15 movies
  q_movies[['title', 'vote_count', 'vote_average', 'score','Poster']]
  return q_movies

  def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


    #Return empty list in case of missing/malformed data
    return []
def url_to_image(url):
  image=""
  resp=""
  try:
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    url1=url
  except Exception as e:
    resp = urllib.request.urlopen("https://www.my24erica.com/assets/images/imdbnoimage.jpg")
    url1="https://www.my24erica.com/assets/images/imdbnoimage.jpg"
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  return url1

  def clean_data(text):
    if isinstance(text, list):
      return [str.lower(i.replace(" ", "")) for i in text]

      
    else:
        #Check if director exists. If not, return empty string
        if isinstance(text, str):
          text=str.lower(text)
          text=re.sub("-","",text)
          text=re.sub(":","",text)
          text=re.sub("'","",text)
          text=re.sub(" ","",text)
          return text
        else:
            return ''

def content_filtering(data1):
  def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
  
  features = ['cast', 'crew', 'keywords', 'genres']
  for feature in features:
    data1[feature] = data1[feature].apply(literal_eval)
  data1['director'] = data1['crew'].apply(get_director)

  features = ['cast', 'keywords', 'genres']
  for feature in features:
    data1[feature] = data1[feature].apply(get_list)
  features = ['cast', 'keywords', 'director', 'genres']

  for feature in features:
    data1[feature] = data1[feature].apply(clean_data)
  data1['soup'] = data1.apply(create_soup, axis=1)
  count = CountVectorizer(stop_words='english')
  count_matrix = count.fit_transform(data1['soup'])
  
  cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
  
  return cosine_sim2         


# Getting data
#Run only when want to prepare data from scratch
data1=prepare_data()
# data1=pd.read_csv('dataset.csv')
# Getting Demographic filtering results
data=Demographic_filtering(data1)
# List of cosine similarities of movies
cosine_sim2=content_filtering(data1)
data1 = data1.reset_index()
indices = pd.Series(data1.index, index=data1['Title1'])
 
import json
from flask import jsonify

# NAMES=["abc","abcd","abcde","abcdef"]
NAMES=list(data1['original_title'])

@app.route('/autocomplete',methods=['GET'])
def autocomplete():
    search = request.args.get('autocomplete')
    app.logger.debug(search)
    return Response(json.dumps(NAMES), mimetype='application/json')
@app.route('/',methods=['GET','POST'])
def index():
    # form = request.form.get("autocomp")
    form="l"
    images=[]
    title=[]
    mainImg=""
    recomHead="Trending Now"
    message="Enter the Movie name"
    if request.method == 'POST':
        # message = request.form.get("autocomp")
        # form = request.form.get("autocomp")

        message = flask.request.form['autocomp']
        if clean_data(message) in list(data1['Title1']):
            print("YSSSS")
            mov=get_recommendations(clean_data(message),cosine_sim2)
            mainImg=data1[data1['Title1']==clean_data(message)]['Poster'].values[0]
            mainImg=url_to_image(mainImg)
            print("mainImg",str(mainImg))
            for i in range(5):
                img1=data1['Poster'][mov.index[i]]
                url1=url_to_image(img1)
                images.append(url1)
                title.append(data1['original_title'][mov.index[i]])
                recomHead="Your Recomendations are"


  
        else:
            a=data.index
            group_of_items =list(a)             # a sequence or set will work here.
            num_to_select = 5                      # set the number to select here.
            list_of_random_items = random.sample(group_of_items, num_to_select)  
            for i in range(5):
              img=url_to_image(data['Poster'][list_of_random_items[i]])
              images.append(img)
              title.append(data['original_title'][list_of_random_items[i]])
            message="No Movie Found"
        
    else:
        a=data.index
        group_of_items =list(a)             # a sequence or set will work here.
        num_to_select = 5                      # set the number to select here.
        list_of_random_items = random.sample(group_of_items, num_to_select)  
        for i in range(5):
          img=url_to_image(data['Poster'][list_of_random_items[i]])
          images.append(img)
          title.append(data['original_title'][list_of_random_items[i]])

    return render_template('index.html',recomHead=recomHead,mainImg=mainImg,title=title,img=images,message=message)


def get_recommendations(title, cosine_sim):
    # Get the index of the movie that matches the title
    idx=2
    try:
      idx = indices[title]
    
    except Exception as e:
      print(e)
      return idx
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data1['title'].iloc[movie_indices]


if __name__=='__main__':
    app.run(debug=True,threaded=False)
