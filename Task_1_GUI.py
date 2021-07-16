import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from PIL import Image
import streamlit as st

st.title("Study Hours VS Marks Prediction")
nav = st.sidebar.radio("Navigation", ["Home", "About Data", "Prediction"])
df= pd.read_csv(r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
model = LinearRegression()
inputs = df[['Hours']]
targets = df.Scores
model.fit(inputs, targets)
if nav == "Home":

    img = Image.open("study.jpg")
    st.image(img, width=700)
    st.write('The data is a record of students with their study hours and scores. We need to make a model for '
             'prediction '
             'of scores based on number of study hours.')

    graph= st.selectbox("What kind of graph?", ["Interactive"])

    if graph=="Interactive":
        layout= go.Layout(
            xaxis=dict(range=[0, 16]),
            yaxis=dict(range=[0, 210000])
        )
        fig= go.Figure(data=go.Scatter(x=df['Hours'], y=df["Scores"], mode='markers'))
        fig.update_layout(
            title="Hours vs Score",
            xaxis_title="Study Hours",
            yaxis_title="Score"
        )
        st.plotly_chart(fig)


if nav == "About Data":
    if st.checkbox("Show Data"):
        if st.checkbox("Head"):
            st.write("Top Five rows")
            st.table(df.head())
        if st.checkbox("Tail"):
            st.write("Last Five rows")
            st.table(df.tail())
    if st.checkbox('Null Values'):
        st.write("Null Values")
        st.write(df.isnull().sum())
    if st.checkbox("Data Description"):
        st.write(df.describe())
    if st.checkbox("Coorelation"):
        st.write(df.Scores.corr(df.Hours))
        st.text("0.97 indicates a very strong coorelation among columns")

if nav == 'Prediction':
    val= st.number_input("Enter Study Hour", 0.00, 10.00, step=0.25)
    predict= model.predict([[val]])

    if st.button("Predict"):
        st.success(f"Predicted Scores is {predict}")

