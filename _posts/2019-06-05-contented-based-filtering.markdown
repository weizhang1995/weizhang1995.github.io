---
layout:     post
title:      "Content Based Filtering - TensorFlow"
subtitle:   "简单的协同过滤推荐算法"
date:       2019-06-05
author:     "Wei Zhang"
header-img: "img/bg/post-2019-Content-Based.jpg"
catalog: true
tags:
    - 推荐系统
    - 复杂网络
    - 线性代数
    - TensorFlow  

---

## Content Based Filtering by hand

This blog illustrates how to implement a content based filter using low level Tensorflow operations. 

(link: https://github.com/GoogleCloudPlatform/training-data-analyst)

To begin with, we need to use TensorFlow version 1.13.1. 


```python
!pip install tensorflow==1.13.1
```

Make sure to restart your kernel to ensure this change has taken place.


```python
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()
print(tf.__version__)
```

out

    1.13.1


To start, we'll create our list of users, movies and features. While the users and movies represent elements in our database, for a content-based filtering method the features of the movies are likely hand-engineered and rely on domain knowledge to provide the best embedding space. Here we use the categories of Action, Sci-Fi, Comedy, Cartoon, and Drama to describe our movies (and thus our users).

In this example, we will assume our database consists of four users and six movies, listed below.  


```python
users = ['Ryan', 'Danielle',  'Vijay', 'Chris']
movies = ['Star Wars', 'The Dark Knight', 'Shrek', 'The Incredibles', 'Bleu', 'Memento']
features = ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']

num_users = len(users)
num_movies = len(movies)
num_feats = len(features)
num_recommendations = 2
```

### Initialize our users, movie ratings and features

We'll need to enter the user's movie ratings and the k-hot encoded movie features matrix. Each row of the users_movies matrix represents a single user's rating (from 1 to 10) for each movie. A zero indicates that the user has not seen/rated that movie. The movies_feats matrix contains the features for each of the given movies. Each row represents one of the six movies, the columns represent the five categories. A one indicates that a movie fits within a given genre/category. 


```python
# each row represents a user's rating for the different movies
users_movies = tf.constant([
                [4,  6,  8,  0, 0, 0],
                [0,  0, 10,  0, 8, 3],
                [0,  6,  0,  0, 3, 7],
                [10, 9,  0,  5, 0, 2]],dtype=tf.float32)

# features of the movies one-hot encoded
# e.g. columns could represent ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']
movies_feats = tf.constant([
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1]],dtype=tf.float32)
```

### Computing the user feature matrix

We will compute the user feature matrix; that is, a matrix containing each user's embedding in the five-dimensional feature space. 


```python
users_feats = tf.matmul(users_movies,movies_feats)
users_feats
```


out

    <tf.Tensor: id=2, shape=(4, 5), dtype=float32, numpy=
    array([[10., 10.,  8.,  8.,  4.],
           [ 3.,  0., 10., 10., 11.],
           [13.,  6.,  0.,  0., 10.],
           [26., 19.,  5.,  5., 12.]], dtype=float32)>



Next we normalize each user feature vector to sum to 1. Normalizing isn't strictly neccesary, but it makes it so that rating magnitudes will be comparable between users.


```python
users_feats = users_feats/tf.reduce_sum(users_feats,axis=1,keepdims=True)
users_feats
```


out

    <tf.Tensor: id=6, shape=(4, 5), dtype=float32, numpy=
    array([[0.25      , 0.25      , 0.2       , 0.2       , 0.1       ],
           [0.0882353 , 0.        , 0.29411766, 0.29411766, 0.32352942],
           [0.44827586, 0.20689656, 0.        , 0.        , 0.3448276 ],
           [0.3880597 , 0.2835821 , 0.07462686, 0.07462686, 0.17910448]],
          dtype=float32)>



### Ranking feature relevance for each user

We can use the users_feats computed above to represent the relative importance of each movie category for each user. 


```python
top_users_features = tf.nn.top_k(users_feats, num_feats)[1]
top_users_features
```


out

    <tf.Tensor: id=10, shape=(4, 5), dtype=int32, numpy=
    array([[0, 1, 2, 3, 4],
           [4, 2, 3, 0, 1],
           [0, 4, 1, 2, 3],
           [0, 1, 4, 2, 3]], dtype=int32)>



#### tf.nn.top_k(input,k)
这个函数等同于 tf.math.top_k(),把input里面前k大的value及其index输出。

tf.nn.top_k(input,k).values =  tf.nn.top_k(input,k)[0]  输出input里面前k大的value

tf.nn.top_k(input,k).indices =  tf.nn.top_k(input,k)[1]  输出input里面前k大的value的index



```python
for i in range(num_users):
    feature_names = [features[int(index)] for index in top_users_features[i]]
    print('{}: {}'.format(users[i],feature_names))
```

out

    Ryan: ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']
    Danielle: ['Drama', 'Comedy', 'Cartoon', 'Action', 'Sci-Fi']
    Vijay: ['Action', 'Drama', 'Sci-Fi', 'Comedy', 'Cartoon']
    Chris: ['Action', 'Sci-Fi', 'Drama', 'Comedy', 'Cartoon']


### Determining movie recommendations. 

We'll now use the `users_feats` tensor we computed above to determine the movie ratings and recommendations for each user.

To compute the projected ratings for each movie, we compute the similarity measure between the user's feature vector and the corresponding movie feature vector.  

We will use the dot product as our similarity measure. In essence, this is a weighted movie average for each user.


```python
users_ratings = tf.matmul(users_feats,tf.transpose(movies_feats))
users_ratings
```


out

    <tf.Tensor: id=130, shape=(4, 6), dtype=float32, numpy=
    array([[0.6       , 0.5       , 0.4       , 0.65      , 0.1       ,
            0.35      ],
           [0.4117647 , 0.0882353 , 0.5882353 , 0.67647064, 0.32352942,
            0.4117647 ],
           [1.        , 0.6551724 , 0.        , 0.44827586, 0.3448276 ,
            0.79310346],
           [0.8507463 , 0.6716418 , 0.14925373, 0.53731346, 0.17910448,
            0.5671642 ]], dtype=float32)>



#### tf.matmul(A, B)
矩阵乘法

#### tf.multiply(A, B)
对应元素相乘

The computation above finds the similarity measure between each user and each movie in our database. To focus only on the ratings for new movies, we apply a mask to the all_users_ratings matrix.  

If a user has already rated a movie, we ignore that rating. This way, we only focus on ratings for previously unseen/unrated movies.


```python
users_ratings_new = tf.where(tf.equal(users_movies, tf.zeros_like(users_movies)),
                                  users_ratings,
                                  tf.zeros_like(tf.cast(users_movies, tf.float32)))
users_ratings_new
```


out

    <tf.Tensor: id=135, shape=(4, 6), dtype=float32, numpy=
    array([[0.        , 0.        , 0.        , 0.65      , 0.1       ,
            0.35      ],
           [0.4117647 , 0.0882353 , 0.        , 0.67647064, 0.        ,
            0.        ],
           [1.        , 0.        , 0.        , 0.44827586, 0.        ,
            0.        ],
           [0.        , 0.        , 0.14925373, 0.        , 0.17910448,
            0.        ]], dtype=float32)>



#### tf.equal(A,B)
用于比较A与B的对应元素是否相同，相同位置上的元素相等则True，否则False。该函数返回值是对应元素True或False的与A、B维度相同的tensor。

#### tf.where(condition，A，B)
condition是bool型值，True/False，A和B维度相同。该函数返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素。x只负责对应替换True的元素，y只负责对应替换False的元素。


Finally let's grab and print out the top 2 rated movies for each user

First step: print the indices of top 2 rated movies for each user.

```python
top_movies = tf.nn.top_k(users_ratings_new, num_recommendations)[1]
top_movies
```


out

    <tf.Tensor: id=139, shape=(4, 2), dtype=int32, numpy=
    array([[3, 5],
           [3, 0],
           [0, 3],
           [4, 2]], dtype=int32)>


And then print out the top 2 rated movies' name.

```python
for i in range(num_users):
    movie_names = [movies[index] for index in top_movies[i]]
    print('{}: {}'.format(users[i],movie_names))
```

out

    Ryan: ['The Incredibles', 'Memento']
    Danielle: ['The Incredibles', 'Star Wars']
    Vijay: ['Star Wars', 'The Incredibles']
    Chris: ['Bleu', 'Shrek']

