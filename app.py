import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

st.title("🌸 Iris Dataset Visualizations")

# Scatter plot Sepal
st.subheader("Sepal Length vs Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=ax)
st.pyplot(fig)

# Scatter plot Petal
st.subheader("Petal Length vs Petal Width")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=ax)
st.pyplot(fig)

# Histogram
st.subheader("Distribution of Sepal Length")
fig, ax = plt.subplots()
sns.histplot(df["sepal length (cm)"], bins=20, kde=True, ax=ax)
st.pyplot(fig)
