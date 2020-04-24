# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.metrics import accuracy_score
from IPython import get_ipython

# %%
import pandas as pd  # Dataframe, Series
import numpy as np  # Scientific computing package, Array
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

import graphviz
import pydotplus
import io
import imageio
from scipy import misc

get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Spotify Song Attributes EDA
# • Import dataset
# • EDA to visualize data and observe structure
# • Train a classifier (Decision Tree)
# • Predict target using the trained classifier

# %%
data = pd.read_csv('data/data.csv')
data.describe()


# %%
data.head()


# %%
data.info()


# %%
# Split data into testing set and training set
train, test = train_test_split(data, test_size=0.15)
# Prints training and test sizes
print(f'Training size: {len(train)}; Test size: {len(test)}')


# %%
# Prints number of rows and columns in data set
train.shape


# %%
# Factors that measure if the user likes/dislikes is taken from the 'target' section of the dataset
# If the target value == 1, select tempo(user liked the song), if the target value == 0(user disliked the song)
# Variables that correspond to factors determining whether or not the user liked the song
pos_tempo = data[data['target'] == 1]['tempo']
neg_tempo = data[data['target'] == 0]['tempo']
pos_dance = data[data['target'] == 1]['danceability']
neg_dance = data[data['target'] == 0]['danceability']
pos_duration = data[data['target'] == 1]['duration_ms']
neg_duration = data[data['target'] == 0]['duration_ms']
pos_loudness = data[data['target'] == 1]['loudness']
neg_loudness = data[data['target'] == 0]['loudness']
pos_speechiness = data[data['target'] == 1]['speechiness']
neg_speechiness = data[data['target'] == 0]['speechiness']
pos_valence = data[data['target'] == 1]['valence']
neg_valence = data[data['target'] == 0]['valence']
pos_energy = data[data['target'] == 1]['energy']
neg_energy = data[data['target'] == 0]['energy']
pos_acousticness = data[data['target'] == 1]['acousticness']
neg_acousticness = data[data['target'] == 0]['acousticness']
pos_key = data[data['target'] == 1]['key']
neg_key = data[data['target'] == 0]['key']
pos_instrumentalness = data[data['target'] == 1]['instrumentalness']
neg_instrumentalness = data[data['target'] == 0]['instrumentalness']

fig = plt.figure(figsize=(12, 8))
plt.title('Song Tempo Like / Dislike Ratio')
pos_tempo.hist(alpha=0.7, bins=30, label='positive')
neg_tempo.hist(alpha=0.7, bins=30, label='negative')
plt.legend(loc='upper right')


# %%
fig2 = plt.figure(figsize=(15, 15))

# Danceability  # Favors low danceability
ax3 = fig2.add_subplot(331)
ax3.set_xlabel('Danceability')
ax3.set_ylabel('Count')
ax3.set_title('Song Danceability: Like Distribution')
pos_dance.hist(alpha=0.5, bins=30)
ax4 = fig.add_subplot(331)
neg_dance.hist(alpha=0.5, bins=30)

# Duration  # Favors lower duration
ax5 = fig2.add_subplot(332)
pos_duration.hist(alpha=0.5, bins=30)
ax5.set_xlabel('Duration (ms)')
ax5.set_ylabel('Count')
ax5.set_title('Song Duration: Like Distribution')
ax6 = fig.add_subplot(332)
neg_duration.hist(alpha=0.5, bins=30)

# Loudness  #Favors low loudness
ax7 = fig2.add_subplot(333)
pos_loudness.hist(alpha=0.5, bins=30)
ax7.set_xlabel('Loudness')
ax7.set_ylabel('Count')
ax7.set_title('Song Loudness: Like Distribution')
ax8 = fig.add_subplot(333)
neg_loudness.hist(alpha=0.5, bins=30)

# Speechiness  # Favors lower speechiness
ax9 = fig2.add_subplot(334)
pos_speechiness.hist(alpha=0.5, bins=30)
ax9.set_xlabel('Speechiness')
ax9.set_ylabel('Count')
ax9.set_title('Song Speechiness: Like Distribution')
ax10 = fig.add_subplot(334)
neg_speechiness.hist(alpha=0.5, bins=30)

# Valence  # Favors high valence
ax11 = fig2.add_subplot(335)
pos_valence.hist(alpha=0.5, bins=30)
ax11.set_xlabel('Valence')
ax11.set_ylabel('Count')
ax11.set_title('Song Valence: Like Distribution')
ax12 = fig.add_subplot(335)
neg_valence.hist(alpha=0.5, bins=30)

# Energy  #Slightly avors low energy
ax13 = fig2.add_subplot(336)
pos_energy.hist(alpha=0.5, bins=30)
ax13.set_xlabel('Energy')
ax13.set_ylabel('Count')
ax13.set_title('Song Energy: Like Distribution')
ax14 = fig.add_subplot(336)
neg_energy.hist(alpha=0.5, bins=30)

# Acousticness  #Favors lower acousticness
ax16 = fig2.add_subplot(338)
pos_acousticness.hist(alpha=0.5, bins=30)
ax16.set_xlabel('Acousticness')
ax16.set_ylabel('Count')
ax16.set_title('Song Acousticness: Like Distribution')
ax16 = fig.add_subplot(338)
neg_acousticness.hist(alpha=0.5, bins=30)

# Key  #Favors keys that aren't so flat
ax15 = fig2.add_subplot(337)
pos_key.hist(alpha=0.5, bins=30)
ax15.set_xlabel('Key')
ax15.set_ylabel('Count')
ax15.set_title('Song Key: Like Distribution')
ax15 = fig.add_subplot(337)
neg_key.hist(alpha=0.5, bins=30)

# Instrumentalness  # Favors lower instrumentalness
ax17 = fig2.add_subplot(339)
pos_instrumentalness.hist(alpha=0.5, bins=30)
ax17.set_xlabel('Instrumentalness')
ax17.set_ylabel('Count')
ax17.set_title('Song Instrumentalness: Like Distribution')
ax17 = fig.add_subplot(339)
neg_instrumentalness.hist(alpha=0.5, bins=30)


# %%
c = DecisionTreeClassifier(min_samples_split=100)


# %%
features = ['danceability', 'loudness', 'valence', 'energy',
            'instrumentalness', 'key', 'speechiness', 'duration_ms']


# %%
x_train = train[features]
y_train = train['target']

x_test = test[features]
y_test = test['target']

# Decision Tree
dt = c.fit(x_train, y_train)


# %%
# Method that visualizes the decision tree using graphiz
def show_tree(tree, features, path):
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.imshow(img)


# %%
show_tree(dt, features, 'dec_tree_01.png')


# %%
# Looks at the test set and ran the decision tree like(1) vs dislike(0) and tested it against the given features until it gets classified as a 1 or 0
y_pred = c.predict(x_test)


# %%
print(y_pred)


# %%
# Tests accuracy by comparing the y_predict list with the y_test list using sklearn

score = accuracy_score(y_test, y_pred) * 100


# %%
print('Accuracy using Decision Tree: ', round(score, 1), '%')
