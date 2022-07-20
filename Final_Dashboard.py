# -*- coding: utf-8 -*-

## FUNDAMENTALS OF DATA SCIENCE
## UNIVERSITY OF APPLIED SCIENCES
## DAAN DEN OTTER
## LECT: JONAS MOONS
## 01-27-2021



    # Default libraries
import json
import pandas as pd
    # Needed for WordCloud
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# Wordcounter
import wordcounter


    # Import Dash and Plotly
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import emoji
from collections import Counter
import regex
import advertools as adv
from textblob import TextBlob



# # # # # # # # # # # # # # # # #

#     D A T A F R A M E S       #

# # # # # # # # # # # # # # # # #

# Creating the mean dataframe
df = pd.read_json("clean_NOS_tweets.json")

# Adding polarity in the dataframe
df['polarity'] = df['text'].str.lower().apply(lambda x: TextBlob(x).sentiment[0] )
# df['polarity'] = round(df['polarity'], 1)

# Adding subjectivity in the dataframe
df['subjectivity'] = df['text'].str.lower().apply(lambda x: TextBlob(x).sentiment[1] )
# df['subjectivity'] = round(df['subjectivity'], 1)

# Dataframe without retweets
df_noRT = df.drop_duplicates(subset =["user_id"])

# Dataframe where ⭕ is used
dfNOS_all = df.loc[(df['text'].str.contains('⭕')) | (df['user_name'].str.contains('⭕')) | (df['user_bio'].str.contains('⭕'))]  # JUST IN CASE
# Dataframe with ⭕ in the name or bio
dfNOS = df.loc[(df['user_name'].str.contains('⭕')) | (df['user_bio'].str.contains('⭕'))]
# Dataframe with 🇳🇱 in the name or bio
dfNL = df.loc[(df['user_name'].str.contains('🇳🇱')) | (df['user_bio'].str.contains('🇳🇱'))]
# Dataframe with 🚫 in the name or bio
dfBLOCK = df.loc[(df['user_name'].str.contains('🚫')) | (df['user_bio'].str.contains('🚫'))]


# EXTRACT EMOJIS

## //// READ THIS!
# ## I cleaned the data in such a way that you had different columns for the emojis.
# ## The emojis were extracted out of the text, username and bio. 
# ## Later I found a workaround and didn't need to use it this way anymore.
# ## I still left in in here for possible future research.
## ////

# ## EMOJIS EXTRACTING TO DATAFRAME

# # extract emojis
# t_emo = adv.extract_emoji(df['text'])

# # the extracted gives a dictionary with all kinds of information. For now, we only need the first column.
# # the first column is a list of the extracted emojis of each row 
# t_emo = t_emo.pop('emoji')

# # Make it a pandas Series so we can add it to the df later.
# t_emo = pd.Series(t_emo, name='text_emojis')

# # now we have a df with one column and as many rows as tweets. We can easily merge it with the existing df.
# df = pd.concat([df, t_emo], axis=1)


# # let's do this also for the emojis in the user names and user bios

# # # User names
# un_emo = adv.extract_emoji(df['user_name'])
# un_emo = un_emo.pop('emoji')
# un_emo = pd.Series(un_emo, name='user_name_emojis')
# df = pd.concat([df, un_emo], axis=1)

# # # Bios
# ub_emo = adv.extract_emoji(df['user_bio'])
# ub_emo = ub_emo.pop('emoji')
# ub_emo = pd.Series(ub_emo, name='user_bio_emojis')
# df = pd.concat([df, ub_emo], axis=1)



# # # # # # # # # # # # # # # # #

#           R O W 1             #
#     COUNTING AND LINE GRAPH   #

# # # # # # # # # # # # # # # # #


# ///// COUNTING FUNCTIONS


# HERE I'M COUNTING THE NUMBERS NEEDED AT THE TOP OF THE DASHBOARD WITH FUNCTIONS

# To count tweets
def countTweets():
    total_tweets = df["text"].count()
    return total_tweets

# To count the users I dropped all the duplicates and counted the different 
def countUsers():
    total_difUsers = len(df.user_id.unique())
    return total_difUsers

# To count the total of verified
def countVerified():
    total_difUsers = df.drop_duplicates(subset =["user_id"])
    total_verified = total_difUsers['verified'].sum()
    return total_verified

# Calculate the amount of tweets that is a retweet 
def countRetweets(): 
    df['retweeted'] = df["text"].str.startswith("RT @" or "rt @")
    retweet_count = sum(df['retweeted'])
    return retweet_count

# Calculate the amount of tweets withouth the retweets, so only original tweets
def countRealTweets():
    num_retweets = countRetweets()
    num_tweets = countTweets()
    real_tweets = num_tweets - num_retweets
    return real_tweets




# ///// LINE GRAPH | TWEETS BY HOUR


    # HERE I CREATE A NEW DF WITHOUT THE RETWEETS
df['retweeted'] = df["text"].str.startswith("RT @" or "rt @")
df_realTweets = df[df["retweeted"] == 0]

    # HERE I GROUP THE NORMAL DF AND THE NEW ONE BY HOURS AND COUNT, I ALSO DO THIS WITH NO RETWEET DF AND EMOJI DF
df_daily_tweets = df.groupby(pd.Grouper(key="tweet_time", freq="H"))["text"].count().reset_index(name="count")
df_daily_retweets = df_realTweets.groupby(pd.Grouper(key="tweet_time", freq="H"))["text"].count().reset_index(name="count")
df_daily_NOS = dfNOS_all.groupby(pd.Grouper(key="tweet_time", freq="H"))["text"].count().reset_index(name="count")


    # CREATING THE FIGURE AND ADDING THE OTHER LINES
df_daily_fig = go.Figure(layout=dict(title=dict(text="Tweets with hashtag #NOS per hour") , xaxis=dict(title='Time (hour)'), yaxis=dict(title='Frequency')))
df_daily_fig = df_daily_fig.add_scatter(x=df_daily_tweets['tweet_time'], y=df_daily_tweets['count'], name="All Tweets")
df_daily_fig = df_daily_fig.add_scatter(x=df_daily_retweets['tweet_time'], y=df_daily_retweets['count'], name="Original Tweets (w/o RT)")
df_daily_fig = df_daily_fig.add_scatter(x=df_daily_NOS['tweet_time'], y=df_daily_NOS['count'], name="Tweets containing ⭕")




# # # # # # # # # # # # # # # # #

#          R O W  2             #
#    EMOJIS AND WORDCLOUD       #

# # # # # # # # # # # # # # # # #


# ///// TOP EMOJIS BAR GRAPH


    # CREATING A FUNCTION THAT EXTRACTS ALL THE EMOJI'S AND COUNTS THEM
def get_emojis(dfcolumn):
    t_emojis = adv.extract_emoji(dfcolumn)
    t_emojis = t_emojis.pop('top_emoji')
    return t_emojis

    # CREATING NEW DATAFRAMES TO EXTRACT EMOJIS OUT OF THE TWEETS, USER NAMES AND BIOS
# Tweets / text
top_emojis_text = pd.DataFrame(get_emojis(df['text']), columns=['Emoji', 'Text_Count'])
# User names
top_emojis_user_name = pd.DataFrame(get_emojis(df['user_name']), columns=['Emoji', 'Name_Count'])
# Bios
top_emojis_user_bio = pd.DataFrame(get_emojis(df['user_bio']), columns=['Emoji', 'Bio_Count'])

    # MERGE THE 3 DATAFRAMES INTO ONE
# Merging the first two dataframes
df_top_emojis = pd.merge(top_emojis_text, top_emojis_user_name, on="Emoji") # To show all emoji's used add : how = 'outer'
# Adding the third
df_top_emojis = pd.merge(df_top_emojis, top_emojis_user_bio, on="Emoji")
# fill na
df_top_emojis = df_top_emojis.fillna(0)
# Add a new column which counts all the columns so you get the total amount
df_top_emojis['Total_Count'] = df_top_emojis['Text_Count'] + df_top_emojis['Name_Count'] + df_top_emojis['Bio_Count']
# Sort the df on the total amount
df_top_emojis = df_top_emojis.sort_values(by="Total_Count", ascending=False)



# ///// WORDCLOUD

    # Setting the first variable and putting all tweets next to each other
words = text = " ".join(review for review in df.text.str.lower())

    # Create stopword list, for weird words I don't want to be in the image
stopwords = set(STOPWORDS)
stopwords.update(["https", "co", "RT"])

    # Add mask, for the image to look like the NOS logo
nos_mask = np.array(Image.open("img/nos_logo.png"))
wc = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", mask=nos_mask).generate(words)
image_colors = ImageColorGenerator(nos_mask)
wc.recolor(color_func=image_colors)

    # Creating the image file in the assets map, so that I can later call the image
wc.to_file("assets/nos_wordcloud.png")

    # To display the generated image:
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()



# # # # # # # # # # # # # # # # #

#           R O W 3             #
#    EMOJI USE & SENTIMENT      #

# # # # # # # # # # # # # # # # #



# ///// EMOJI ANALYSIS


# Get the top emojis from the tweets out of the df you give it
def get_top5_emojis(df_col):
    top_emo = adv.extract_emoji(df_col['text'])
    top_emo = top_emo.pop('top_emoji')
    top_emo = pd.DataFrame(top_emo, columns=['Emoji', 'Count'])
    top_emo = top_emo[top_emo['Count'] > 1]
    return top_emo#.head(5)


# Create tabs to show the different pie charts in
tab_nos = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Emoji use of people with ⭕ in their name or bio'),
            dcc.Graph(figure = px.pie(get_top5_emojis(dfNOS), values='Count', names='Emoji', color_discrete_sequence=px.colors.sequential.RdBu))
        ]
    ),
    className="mt-3",
)

tab_nl = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Emoji use of people with 🇳🇱 in their name or bio'),
            dcc.Graph(figure = px.pie(get_top5_emojis(dfNL), values='Count', names='Emoji', color_discrete_sequence=px.colors.sequential.Reds))
        ]
    ),
    className="mt-3",
)

tab_block = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Emoji use of people with 🚫 in their name or bio'),
            dcc.Graph(figure = px.pie(get_top5_emojis(dfBLOCK), values='Count', names='Emoji', color_discrete_sequence=px.colors.sequential.RdBu))
        ]
    ),
    className="mt-3",
)

# ///// SENTIMENT ANALYSIS
# The histograms are assigned in the dashboard




# # # # # # # # # # # # # # # # # #

# # D A S H B O A R D             #

# # # # # # # # # # # # # # # # # #



# create app object
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# assign layout
app.layout = html.Div(children = [
    # HEADER
    html.Div([
        html.Img(src=app.get_asset_url('nos_news.jpg'), className="header_image")
        ]),
    dbc.Row([
        dbc.Col([
            html.H2("NOS obtains logo from satellite vehicles after persistent threats"),
            html.P("On October 15th 2020 the Dutch mainstream news broadcaster NOS published an article that said they had to remove the NOS logo of their busses because of various attacks on them. These attacks were mostly from right wing extremists who believe that the NOS broadcasts fake news. After the article was published, other people started to show their support on Twitter using the ⭕ emoji as representation of the NOS logo. This dashboard shows the results of all people who used the hashtag #NOS on Twitter in the following week. ")
            ],
            className="pretty_container four columns",
            id="cross-filter-options",
            width=3
            ),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5(countTweets()),
                        html.P("Number of Tweets")
                        ],
                        id="wells",
                        className="mini_container",
                        ),
                    html.Div([
                        html.H5(countUsers()),
                        html.P("Number of Users")
                        ],
                        id="gas",
                        className="mini_container",
                        ),
                    html.Div([
                        html.H5(countVerified()),
                        html.P("Number of Verified Users")],
                        id="oil",
                        className="mini_container",
                        ),
                    ],
                    id="info-container",
                    className="row container-display",
                    ),
                html.Div(
                    dcc.Graph(figure = df_daily_fig),
                    id="countGraphContainer",
                    className="pretty_container",
                    ),
                    ],
                id="right-column",
                className="eight columns",
                ),
            ],
            className="row flex-display",
            ),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H5("Most used emoji's"),
                dcc.Dropdown(
                    id='emoji-dropdown',
                    options=[
                        {'label': 'Total', 'value': 'Total_Count'},
                        {'label': 'Tweets', 'value': 'Text_Count'},
                        {'label': 'User name', 'value': 'User_Count'},
                        {'label': 'Bio', 'value': 'Bio_Count'},                    
                        ],
                    value='Total_Count',
                    ),
                html.Div(id='div-output'),
                dcc.Graph(id = 'example-graph')
                ]),
            className="mini_container",
            width=8
            ),
        dbc.Col(
            html.Div([
                html.H5("Wordcloud of #NOS"),
                html.P('This word cloud is a collection of all the words used in the tweets shown in different sizes. Each size stand for the frequency of the word.'),
                html.Br(),
                html.Img(src=app.get_asset_url('nos_wordcloud.png'), 
                         className="wordcloud_img"),
                ],
                className="mini_container")
            ),
    ]),    
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H5("Which emojis do they use?"),
                html.P("Based on the 3 most used emojis, 3 groups are created based on using one of these emojis in their username or bio. Each tab below stands for one of those groups of Twitter users. The pie chart shows the emojis they use in their tweets. Each emoji must have been used more than 1 time."),
                dbc.Tabs([
                    dbc.Tab(tab_nos, label="⭕"),
                    dbc.Tab(tab_nl, label="🇳🇱"),
                    dbc.Tab(tab_block, label="🚫"),
                ])
                ],
                className="mini_container",)),
        dbc.Col(
            html.Div([
                html.H5("Sentiment analysis"),
                html.P('A sentiment analysis was made for the three groups.'),
                dcc.Dropdown(
                    id='top3-dropdown',
                    options=[
                        {'label': '⭕  (' + str(len(dfNOS)) + ' Tweets)', 'value': 'nos'},
                        {'label': '🇳🇱  (' + str(len(dfNL)) + ' Tweets)', 'value': 'nl'},
                        {'label': '🚫  (' + str(len(dfBLOCK)) + ' Tweets)', 'value': 'block'}                    
                        ],
                    value='nos',
                    # multi=True
                    ),
                html.Br(),
                dcc.RadioItems(
                    id="polsub",
                    options=[
                        {"label":" Polarity   ", "value":"polarity"},
                        {"label":" Subjectivity", "value":"subjectivity"}
                    ],
                    value = "polarity",
                    labelStyle={
                                "display": "inline-block"
                                }
                    ),
                html.Br(),
                html.P('The polarity looks at the positivity of the tweets, in a range from -1,5 (negative) to 1,5 (positive). The subjectivity stands for the emotion of the users in the tweets, ranged between 0 (no emotion) and 1,5 (emotional). The tweets with a score of 0 are left out to have a clearer overview'),
                dcc.Graph(id = 'top3-graph')
                
                ]),
            className="mini_container",
            width=8
            ),
    ]),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('emoji-dropdown', 'value')]
)
def update_output(emojis):

    if 'Text_Count' in emojis:
        df_top_emojis.sort_values(by='Text_Count', ascending=False)
        value = 'Text_Count'#df_top_emojis['Text_Count']
    elif 'User_Count' in emojis:
        df_top_emojis.sort_values(by='Name_Count', ascending=False)
        value = 'Name_Count'#df_top_emojis['Name_Count'] 
    elif 'Bio_Count' in emojis:
        df_top_emojis.sort_values(by='Bio_Count', ascending=False)
        value = 'Bio_Count'#df_top_emojis['Bio_Count']
    elif 'Total_Count' in emojis:
        df_top_emojis.sort_values(by='Total_Count', ascending=False)
        value = 'Total_Count'#df_top_emojis['Total_Count']
    
    
    return px.bar(df_top_emojis, x="Emoji", y=value, title='Most used emojis in Tweets/Names/Bios/Total (select above)', labels={value:'Frequency'})

@app.callback(
    dash.dependencies.Output('top3-graph', 'figure'),
    [dash.dependencies.Input('top3-dropdown', 'value'),
     dash.dependencies.Input('polsub', 'value')]
)
def update_top3(emojis, polsub):
    
    what_emoji = emojis
    pol_or_sub = polsub
    labels_var = {'x':pol_or_sub, 'y':'Frequency (%)'}
    
    if 'nos' in what_emoji:
        tit_emo = '⭕'
    elif 'nl' in what_emoji:
        tit_emo = '🇳🇱'
    elif 'block' in what_emoji:
        tit_emo = '🚫'
    
    title_var = 'Histogram of ' + pol_or_sub + ' in ' + tit_emo
    
    if 'nos' in what_emoji:
        dfNOS1 = dfNOS[dfNOS[pol_or_sub] != 0.0]
        fig = px.histogram(dfNOS1[pol_or_sub], title=title_var, labels=labels_var, histnorm='probability')
    elif 'nl' in what_emoji:
        dfNL1 = dfNL[dfNL[pol_or_sub] != 0.0]
        fig = px.histogram(dfNL1[pol_or_sub], title=title_var, labels=labels_var, histnorm='probability')
    elif 'block' in what_emoji:
        dfBL1 = dfBLOCK[dfBLOCK[pol_or_sub] != 0.0]
        fig = px.histogram(dfBL1[pol_or_sub], title=title_var, labels=labels_var, histnorm='probability')
    
    fig.update_layout(
        xaxis_title_text=pol_or_sub, # xaxis label
        yaxis_title_text='Frequency', # yaxis label
        showlegend=False)
    
    return fig
    


# run the server when executed
if __name__ == "__main__":
  app.run_server(debug=True)