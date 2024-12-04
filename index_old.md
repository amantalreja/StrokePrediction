# **Analysis of Stroke Prediction in Adults🫀🩺🧠**
### Fall 2024 Data Science Project
### Sathvika Sangoju, Aman Talreja, and Aesha Kapoor
### Contributions:

* **A: Project idea** - Everyone contributed as we all spent around 20 minutes trying to find different datasets and then narrowed it down to one that we had a lot of data for and knew we could work with…. (reword)
* **B: Dataset Curation and Preprocessing** - Something similar to A, but describe each part of how we did it as a group. Everyone has different ideas, but we talk about it as a group.
* **C: Data Exploration and Summary Statistics** -
* **D: ML Algorithm Design/Development** -
* **E: ML Algorithm Training and Test Data Analysis** -
* **F: Visualization, Result Analysis, Conclusion** -
* **G: Final Tutorial Report Creation** -
* **H: Additional (not listed above)** - N/A

# **Introduction**
**What was our motivation with this project?** Strokes are a leading cause of disability and mortality worldwide, with over 25% of adults experiencing a stroke at least once in their lifetime. This alarming statistic highlights the pervasive nature of strokes as a medical issue affecting millions globally. Our project aims to address this critical public health concern by analyzing various factors that contribute to stroke risk. By developing predictive insights, we hope to empower individuals and healthcare professionals to take preventative measures and mitigate the likelihood of strokes, ultimately improving health outcomes.

**What question(s) does our analysis answer?** The primary question we aim to answer is: “What factors significantly contribute to an individual’s risk of experiencing a stroke, and how can we predict this risk effectively?”  We are interested in exploring demographic factors such as marital status and how they might influence stroke risk, as well as lifestyle-related variables, including the effects of one’s work environment. In addition, we plan to investigate critical health indicators, such as BMI, hypertension, and smoking habits, to better understand their impact on stroke occurrence. By diving deeper into these and other variables like age, physical activity, and medical history, we aim to uncover patterns and correlations that can provide actionable insights into stroke prevention.

**Why is answering those questions important?** Understanding the factors that contribute to stroke risk is essential for early identification and prevention. Answering these questions will enable individuals to make informed decisions and adopt healthier lifestyles to reduce their stroke risk. For healthcare providers, our analysis can guide personalized recommendations, improve patient care, and help allocate resources more effectively. By understanding one’s risk level for a potential stroke, individuals can take proactive measures to protect themselves, potentially avoiding life-threatening consequences.



# **Data Curation**

**Source of Data:**
*   Dataset Source: Kaggle || Owner: fedesoriano (username on Kaggle)
* Link to dataset on Kaggle:https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download

**What does this data mean?** The dataset we selected is highly relevant to the topic of stroke prevention and provides valuable information for predicting stroke risk. It is well-organized and contains 12 key factors across 5,110 rows of data, offering a robust foundation for our analysis. The dataset includes variables such as marital status and work environment, which may play a role in determining an individual’s likelihood of experiencing a stroke. Additionally, it captures critical health and lifestyle indicators like BMI, hypertension, smoking habits, and more. By analyzing these factors, we aim to uncover meaningful patterns that can help people take preventative measures and reduce their risk of stroke.

**Data Transformation**:
In this stage, we will clean and prepare the dataset to ensure it is suitable for the exploratoy data analysis and primary analysis stages. This will envolve handling missing values by removing rows or columns containing NaN or Unknown values, depending on their significance and frequency. Additionally, categorical variables will be converted into numerical representations using one-hot encoding, allowing them to be used effectively in machine learning models. These transformations will help standardize the data and ensure consistency, making it ready for exploratory analysis and subsequent modeling steps.


```python
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn as sk
```


```python
# Reading in .csv file
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head(20)
#df.shape (shape is (5110, 12))
```





  <div id="df-8500c3bf-b7df-4322-9d7f-234d35aeefc6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9046</td>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51676</td>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31112</td>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60182</td>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1665</td>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>56669</td>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>53882</td>
      <td>Male</td>
      <td>74.0</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>70.09</td>
      <td>27.4</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10434</td>
      <td>Female</td>
      <td>69.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>94.39</td>
      <td>22.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>27419</td>
      <td>Female</td>
      <td>59.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>76.15</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>60491</td>
      <td>Female</td>
      <td>78.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>58.57</td>
      <td>24.2</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12109</td>
      <td>Female</td>
      <td>81.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>80.43</td>
      <td>29.7</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12095</td>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>120.46</td>
      <td>36.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12175</td>
      <td>Female</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>104.51</td>
      <td>27.3</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8213</td>
      <td>Male</td>
      <td>78.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>219.84</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5317</td>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>214.09</td>
      <td>28.2</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>58202</td>
      <td>Female</td>
      <td>50.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>167.41</td>
      <td>30.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>56112</td>
      <td>Male</td>
      <td>64.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>191.61</td>
      <td>37.5</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>34120</td>
      <td>Male</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>221.29</td>
      <td>25.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>27458</td>
      <td>Female</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>89.22</td>
      <td>37.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25226</td>
      <td>Male</td>
      <td>57.0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>217.08</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8500c3bf-b7df-4322-9d7f-234d35aeefc6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8500c3bf-b7df-4322-9d7f-234d35aeefc6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8500c3bf-b7df-4322-9d7f-234d35aeefc6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-75a6f426-9471-4bf7-a712-874bf377f3ad">
  <button class="colab-df-quickchart" onclick="quickchart('df-75a6f426-9471-4bf7-a712-874bf377f3ad')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-75a6f426-9471-4bf7-a712-874bf377f3ad button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Dropping the "id" column
df = df.drop('id', axis=1)
df.head(20)
#df.shape (shape is (5110, 11))
```





  <div id="df-2c742760-ad21-4389-9255-404696217987" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>202.21</td>
      <td>NaN</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Male</td>
      <td>74.0</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>70.09</td>
      <td>27.4</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>69.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>94.39</td>
      <td>22.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Female</td>
      <td>59.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>76.15</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Female</td>
      <td>78.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>58.57</td>
      <td>24.2</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Female</td>
      <td>81.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>80.43</td>
      <td>29.7</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>120.46</td>
      <td>36.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>104.51</td>
      <td>27.3</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Male</td>
      <td>78.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>219.84</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>214.09</td>
      <td>28.2</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Female</td>
      <td>50.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>167.41</td>
      <td>30.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Male</td>
      <td>64.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>191.61</td>
      <td>37.5</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Male</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>221.29</td>
      <td>25.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Female</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>89.22</td>
      <td>37.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Male</td>
      <td>57.0</td>
      <td>0</td>
      <td>1</td>
      <td>No</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>217.08</td>
      <td>NaN</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2c742760-ad21-4389-9255-404696217987')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2c742760-ad21-4389-9255-404696217987 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2c742760-ad21-4389-9255-404696217987');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fc413dc3-21dc-4af0-9bb2-a4e41dba447a">
  <button class="colab-df-quickchart" onclick="quickchart('df-fc413dc3-21dc-4af0-9bb2-a4e41dba447a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fc413dc3-21dc-4af0-9bb2-a4e41dba447a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Dropped the NaN values in the "bmi" column
df = df.dropna(subset=['bmi'])
df.head(20)
#df.shape (shape is (4909, 11))
```





  <div id="df-351cc68c-7a8e-4116-9bd4-daed3d0a51c5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Male</td>
      <td>74.0</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>70.09</td>
      <td>27.4</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>69.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>94.39</td>
      <td>22.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Female</td>
      <td>78.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>58.57</td>
      <td>24.2</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Female</td>
      <td>81.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>80.43</td>
      <td>29.7</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>120.46</td>
      <td>36.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>104.51</td>
      <td>27.3</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>214.09</td>
      <td>28.2</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Female</td>
      <td>50.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>167.41</td>
      <td>30.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Male</td>
      <td>64.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>191.61</td>
      <td>37.5</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Male</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>221.29</td>
      <td>25.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Female</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>89.22</td>
      <td>37.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Female</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>193.94</td>
      <td>22.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Female</td>
      <td>52.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>233.29</td>
      <td>48.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>228.70</td>
      <td>26.6</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Male</td>
      <td>82.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>208.30</td>
      <td>32.5</td>
      <td>Unknown</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-351cc68c-7a8e-4116-9bd4-daed3d0a51c5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-351cc68c-7a8e-4116-9bd4-daed3d0a51c5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-351cc68c-7a8e-4116-9bd4-daed3d0a51c5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-834383ef-413c-4973-9c3d-8ad51372c9ea">
  <button class="colab-df-quickchart" onclick="quickchart('df-834383ef-413c-4973-9c3d-8ad51372c9ea')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-834383ef-413c-4973-9c3d-8ad51372c9ea button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Dropped the "Unknown" values in the "smoking_status" column
df = df[df['smoking_status'] != 'Unknown']
df.head(20)
#df.shape (shape is (3426, 11))
```





  <div id="df-fa88f635-b4ff-4ac2-b0e4-6e15cbcc1acb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>67.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>228.69</td>
      <td>36.6</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>105.92</td>
      <td>32.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>171.23</td>
      <td>34.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>79.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>174.12</td>
      <td>24.0</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>81.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>186.21</td>
      <td>29.0</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Male</td>
      <td>74.0</td>
      <td>1</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>70.09</td>
      <td>27.4</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>69.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>94.39</td>
      <td>22.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Female</td>
      <td>81.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>80.43</td>
      <td>29.7</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Female</td>
      <td>61.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>120.46</td>
      <td>36.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>104.51</td>
      <td>27.3</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>214.09</td>
      <td>28.2</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Female</td>
      <td>50.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>167.41</td>
      <td>30.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Male</td>
      <td>64.0</td>
      <td>0</td>
      <td>1</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>191.61</td>
      <td>37.5</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Male</td>
      <td>75.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>221.29</td>
      <td>25.8</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Female</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Urban</td>
      <td>89.22</td>
      <td>37.8</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Female</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Rural</td>
      <td>193.94</td>
      <td>22.4</td>
      <td>smokes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Female</td>
      <td>52.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>233.29</td>
      <td>48.9</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Female</td>
      <td>79.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>228.70</td>
      <td>26.6</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>102.87</td>
      <td>27.2</td>
      <td>formerly smoked</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Rural</td>
      <td>104.12</td>
      <td>23.5</td>
      <td>never smoked</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fa88f635-b4ff-4ac2-b0e4-6e15cbcc1acb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-fa88f635-b4ff-4ac2-b0e4-6e15cbcc1acb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fa88f635-b4ff-4ac2-b0e4-6e15cbcc1acb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-0842b33a-b92f-484e-a681-651a758aacec">
  <button class="colab-df-quickchart" onclick="quickchart('df-0842b33a-b92f-484e-a681-651a758aacec')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-0842b33a-b92f-484e-a681-651a758aacec button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# One hot encoded all cateogircal vaeribale columns
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
df = df.astype(int)
df.head(20)
#df.shape (shape is (3426, 21))
```





  <div id="df-17ab8c1c-90cd-4914-b87e-11edc3851616" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
      <th>gender_Female</th>
      <th>gender_Male</th>
      <th>gender_Other</th>
      <th>ever_married_No</th>
      <th>...</th>
      <th>work_type_Govt_job</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Rural</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>0</td>
      <td>1</td>
      <td>228</td>
      <td>36</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>80</td>
      <td>0</td>
      <td>1</td>
      <td>105</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>171</td>
      <td>34</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79</td>
      <td>1</td>
      <td>0</td>
      <td>174</td>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>81</td>
      <td>0</td>
      <td>0</td>
      <td>186</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>74</td>
      <td>1</td>
      <td>1</td>
      <td>70</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>69</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>81</td>
      <td>1</td>
      <td>0</td>
      <td>80</td>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>61</td>
      <td>0</td>
      <td>1</td>
      <td>120</td>
      <td>36</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>79</td>
      <td>0</td>
      <td>1</td>
      <td>214</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>167</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>64</td>
      <td>0</td>
      <td>1</td>
      <td>191</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>75</td>
      <td>1</td>
      <td>0</td>
      <td>221</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>89</td>
      <td>37</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>193</td>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>233</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>79</td>
      <td>0</td>
      <td>0</td>
      <td>228</td>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>102</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>80</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-17ab8c1c-90cd-4914-b87e-11edc3851616')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-17ab8c1c-90cd-4914-b87e-11edc3851616 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-17ab8c1c-90cd-4914-b87e-11edc3851616');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6125e551-ea17-460e-ad8c-c0cf1e5f2784">
  <button class="colab-df-quickchart" onclick="quickchart('df-6125e551-ea17-460e-ad8c-c0cf1e5f2784')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6125e551-ea17-460e-ad8c-c0cf1e5f2784 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# **Exploratory Data Analysis**

In this stage, we aim to thoroughly explore our dataset to uncover patterns, relationships, and potential issues that could influence our modeling process. This involves taking a closer look at the structure and distribution of the data to gain valuable insights. Throughout this section, we will draw conclusions based on the data and test whether they hold true, using visualizations and statistical summaries to validate our findings.


**CONCLUSION 1: ARE THERE OUTLIERS?**


```python
# Setting the columns we want to view
columns_to_plot = ['age', 'avg_glucose_level', 'bmi']
df_to_plot = df[columns_to_plot]
# Box plot ot visaulize data for any outliers
plt.figure(figsize=(10, 8))
df_to_plot.boxplot()
plt.title('Box Plots for Age, BMI, and Avg. Glucose Level')
plt.ylabel('Values')
plt.grid(True)
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_11_0.png)
    


**CONCLUSION 1 ANALYSIS:** Based on the visualization above, we can see that there the data contains outliers. Since there are no NaN values in these columns, we can standarduze the data instead of normalizing the data


```python
# We are stadardizing the column values for "age", "bmi", and "avg_glucose_level" to scale the data by converting the values to z-scores because they ar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'bmi', 'avg_glucose_level']] = scaler.fit_transform(df[['age', 'bmi', 'avg_glucose_level']])
#pd.set_option('display.max_columns', None)
df.head(20)
```





  <div id="df-7ecff486-b2e2-4773-873c-92ebb52bb3cc" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
      <th>gender_Female</th>
      <th>gender_Male</th>
      <th>gender_Other</th>
      <th>ever_married_No</th>
      <th>...</th>
      <th>work_type_Govt_job</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
      <th>Residence_type_Rural</th>
      <th>Residence_type_Urban</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.973768</td>
      <td>0</td>
      <td>1</td>
      <td>2.519900</td>
      <td>0.844845</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.663479</td>
      <td>0</td>
      <td>1</td>
      <td>-0.059459</td>
      <td>0.295850</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.018784</td>
      <td>0</td>
      <td>0</td>
      <td>1.324587</td>
      <td>0.570347</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.610424</td>
      <td>1</td>
      <td>0</td>
      <td>1.387499</td>
      <td>-0.802140</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.716533</td>
      <td>0</td>
      <td>0</td>
      <td>1.639143</td>
      <td>-0.115896</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.345151</td>
      <td>1</td>
      <td>1</td>
      <td>-0.793423</td>
      <td>-0.390394</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.079877</td>
      <td>0</td>
      <td>0</td>
      <td>-0.290133</td>
      <td>-1.076637</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.716533</td>
      <td>1</td>
      <td>0</td>
      <td>-0.583719</td>
      <td>-0.115896</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.655440</td>
      <td>0</td>
      <td>1</td>
      <td>0.255097</td>
      <td>0.844845</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.284058</td>
      <td>0</td>
      <td>0</td>
      <td>-0.080429</td>
      <td>-0.390394</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.610424</td>
      <td>0</td>
      <td>1</td>
      <td>2.226314</td>
      <td>-0.253145</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.071839</td>
      <td>1</td>
      <td>0</td>
      <td>1.240706</td>
      <td>0.021352</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.814604</td>
      <td>0</td>
      <td>1</td>
      <td>1.743995</td>
      <td>0.982093</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.398205</td>
      <td>1</td>
      <td>0</td>
      <td>2.373107</td>
      <td>-0.664891</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.602386</td>
      <td>0</td>
      <td>0</td>
      <td>-0.394985</td>
      <td>0.982093</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.185987</td>
      <td>0</td>
      <td>0</td>
      <td>1.785936</td>
      <td>-1.076637</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.177948</td>
      <td>1</td>
      <td>0</td>
      <td>2.624752</td>
      <td>2.491829</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.610424</td>
      <td>0</td>
      <td>0</td>
      <td>2.519900</td>
      <td>-0.527642</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.185987</td>
      <td>0</td>
      <td>0</td>
      <td>-0.122370</td>
      <td>-0.390394</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.663479</td>
      <td>0</td>
      <td>0</td>
      <td>-0.080429</td>
      <td>-0.939389</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7ecff486-b2e2-4773-873c-92ebb52bb3cc')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7ecff486-b2e2-4773-873c-92ebb52bb3cc button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7ecff486-b2e2-4773-873c-92ebb52bb3cc');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b5572a61-c7f1-4f6e-ba83-db7202b44ee9">
  <button class="colab-df-quickchart" onclick="quickchart('df-b5572a61-c7f1-4f6e-ba83-db7202b44ee9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b5572a61-c7f1-4f6e-ba83-db7202b44ee9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Visualizing data after standardizing it
columns_to_plot = ['age', 'avg_glucose_level', 'bmi']
df_to_plot = df[columns_to_plot]
plt.figure(figsize=(10, 8))
df_to_plot.boxplot()
plt.title('Box Plots for Age, BMI, and Avg. Glucose Level')
plt.ylabel('Values')
plt.grid(True)
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_14_0.png)
    


**CONCLUSION 1 ANALYSIS:** Based on the visualization after standardizing the data, we can see that there are no outliers for the age. However, there are several outliers for avg_glucose_level and bmi. The outliers for avg_glucose_level are closer together, while the outliers for bmi are more spaced out.


**CONCLUSION 2: ARE BMI AND AVG_GLUCOSE_LEVEL STATISTICALLY SIGNIFICANT?**

Our hypothesis with a confidence level of 95% and an alpha value of 0.05:

*   **Null Hypothesis:** The BMI and avg_glucose_level are not correlated
*  **Alternative Hypothesis:** The BMI and avg_glucose_level are correlated




```python
# Making a scatter plot to visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(df['bmi'], df['avg_glucose_level'], alpha=0.5)
plt.title('BMI vs. Avg. Glucose Level')
plt.xlabel('BMI')
plt.ylabel('Avg. Glucose Level')
plt.grid(True)
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_18_0.png)
    



```python
# Performing Pearson Correlation Test
correlation, p_value = stats.pearsonr(df['bmi'], df['avg_glucose_level'])
print(f"Pearson Correlation Coefficient: {correlation}")
print(f"P-value: {p_value}")
```

    Pearson Correlation Coefficient: 0.15611142307377204
    P-value: 3.907066447649802e-20


**CONCLUSION 2 ANALYSIS**: Based on the visualization and the results of the Pearson correlation test, we can see that the relationship between the bmi and avg_glocose_level is a weak, positive, and scattered. Since the p-value is practically 0, we can fail to reject the null hypothesis as the p-value of 0 < 0.05. Therefore, we can conclude that there is not enough evidence to say that there is a significant correlation between bmi and avg_glucose_level.


**CONCLUSION 3: IS THE BMI STATISTICALLY SIGNIFICANT TO WHETHER SOMEONE SMOKES OR NOT?**

Our hypothesis with a confidence level of 95% and an alpha value of 0.05:

*   **Null Hypothesis:** The smoking status has no effect on BMI
*   **Alternative Hypothesis:** The smoking status has an effect on BMI


```python
# Split the data into two groups based on the heart disease column
yes_smoked = df[df['smoking_status_never smoked'] == 0]['bmi']
no_smoked = df[df['smoking_status_never smoked'] == 1]['bmi']

#Import ttest_ind
from scipy.stats import ttest_ind
#Implement the t-test for these two groups
t_stat, p_value = ttest_ind(yes_smoked, no_smoked, equal_var=False)

print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')
```

    T-statistic: 2.689885719263453
    P-value: 0.007182760081744601



```python
#Visualizing the "yes_smoked" and the "no_smoked" with box-plots
df['smoked'] = df['smoking_status_never smoked'].apply(lambda x: 'No' if x == 1 else 'Yes')

plt.figure(figsize=(8, 6))
sns.boxplot(x='smoked', y='bmi', data=df)
plt.title('BMI Distribution by Smoking Status')
plt.xlabel('Smoked')
plt.ylabel('BMI')
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_24_0.png)
    


**CONCLUSION 3 ANALYSIS**: Based on the results of the t-test. We can see that the p-value of 0.007 < the alpha value of 0.05. Therefore, we can reject the null hypothesis and accept the alternative hypothesis. Thus, the smoking status of a person has an effect on their BMI. Even though the visualizations of the box plots look similar, they are not the same. We can see that median BMI value of the people who smoke is slightly higher than the BMI of people who do not smoke. However,in the box plot of people who do not smoke, there are more outliers that are also more spread out comapred to people who do smoke.

# **Primary Analysis**
In this stage, we will use insights from the exploratory data analysis to select and implement a machine learning technique that aligns with the goals outlined in the introduction. Since our objective is to predict and identify factors contributing to outcomes such as stroke occurrence, we will use a supervised learning classification approach, given the categorical nature of the target variable.

We plan on implementing a logistic regression model and evaluationg the model using metrics like accuracy, precision, recall, and F1-score to ensure reliability and effectiveness in addressing the research questions.

Additionally, our goal is to identify the top three factors most strongly associated with stroke. To achieve this, we will use a decision tree to uncover thresholds and conditions that highlight the most significant contributors to stroke risk.


```python
#imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
```


```python
#Dropping the smoked column, remove comments if you run into an error, comment it back and then re reun this cell.
#Comment out after running it once

if 'smoked' in df.columns:
    df = df.drop('smoked', axis=1)
    print(df.columns)
    print(df.dtypes)
    print(df['smoked'].unique())
    print(df.head())
else:
    print()


```

    



```python
# Retrieve the scaler's mean and scale for the columns
mean_age, mean_bmi, mean_glucose = scaler.mean_  # Mean values for the three columns
std_age, std_bmi, std_glucose = scaler.scale_    # Standard deviations for the three columns

# Reverse the standardization for 'age'
df['age_original'] = (df['age'] * std_age) + mean_age

# Reverse the standardization for 'bmi'
df['bmi_original'] = (df['bmi'] * std_bmi) + mean_bmi

# Reverse the standardization for 'avg_glucose_level'
df['avg_glucose_level_original'] = (df['avg_glucose_level'] * std_glucose) + mean_glucose

# Display the first few rows to verify
print(df[['age', 'age_original', 'bmi', 'bmi_original', 'avg_glucose_level', 'avg_glucose_level_original']].head(20))
```

             age  age_original       bmi  bmi_original  avg_glucose_level  \
    0   0.973768          67.0  0.844845          36.0           2.519900   
    2   1.663479          80.0  0.295850          32.0          -0.059459   
    3   0.018784          49.0  0.570347          34.0           1.324587   
    4   1.610424          79.0 -0.802140          24.0           1.387499   
    5   1.716533          81.0 -0.115896          29.0           1.639143   
    6   1.345151          74.0 -0.390394          27.0          -0.793423   
    7   1.079877          69.0 -1.076637          22.0          -0.290133   
    10  1.716533          81.0 -0.115896          29.0          -0.583719   
    11  0.655440          61.0  0.844845          36.0           0.255097   
    12  0.284058          54.0 -0.390394          27.0          -0.080429   
    14  1.610424          79.0 -0.253145          28.0           2.226314   
    15  0.071839          50.0  0.021352          30.0           1.240706   
    16  0.814604          64.0  0.982093          37.0           1.743995   
    17  1.398205          75.0 -0.664891          25.0           2.373107   
    18  0.602386          60.0  0.982093          37.0          -0.394985   
    20  1.185987          71.0 -1.076637          22.0           1.785936   
    21  0.177948          52.0  2.491829          48.0           2.624752   
    22  1.610424          79.0 -0.527642          26.0           2.519900   
    24  1.185987          71.0 -0.390394          27.0          -0.122370   
    25  1.663479          80.0 -0.939389          23.0          -0.080429   
    
        avg_glucose_level_original  
    0                        228.0  
    2                        105.0  
    3                        171.0  
    4                        174.0  
    5                        186.0  
    6                         70.0  
    7                         94.0  
    10                        80.0  
    11                       120.0  
    12                       104.0  
    14                       214.0  
    15                       167.0  
    16                       191.0  
    17                       221.0  
    18                        89.0  
    20                       193.0  
    21                       233.0  
    22                       228.0  
    24                       102.0  
    25                       104.0  



```python
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize SGDClassifier with logistic loss
model = SGDClassifier(loss='log_loss', max_iter=100, warm_start=True, random_state=42)

# Track metrics
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# Number of iterations
n_iterations = 100

# Training loop
for iteration in range(n_iterations):
    model.fit(X_train, y_train)  # Incremental fitting with `warm_start=True`

    # Predict probabilities for loss calculation
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)

    # Compute log loss
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)

    # Compute accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Append metrics
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Final evaluation
y_pred = model.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

    Final Accuracy: 0.9154518950437318
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.94      0.97      0.96       642
               1       0.18      0.09      0.12        44
    
        accuracy                           0.92       686
       macro avg       0.56      0.53      0.54       686
    weighted avg       0.89      0.92      0.90       686
    
    Confusion Matrix:
     [[624  18]
     [ 40   4]]



```python
# Get feature names and corresponding coefficients
coefficients = model.coef_[0]
feature_names = X.columns

# Create a DataFrame of features and their coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Add a column for the absolute value of coefficients
importance_df['Absolute_Coefficient'] = np.abs(importance_df['Coefficient'])

# Sort by absolute coefficient values
importance_df = importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

# Get the top 3 factors
top_3_factors = importance_df.head(3)

# Display the top 3 factors
print("All factors: ")
print(importance_df)
print()
print("Top 3 factors: ")
print(top_3_factors)
print()
print(f"The top 3 factors that lead to stroke are: {', '.join(top_3_factors['Feature'].values)}")
```

    All factors: 
                               Feature  Coefficient  Absolute_Coefficient
    20                    age_original   235.422987            235.422987
    21                    bmi_original   109.159248            109.159248
    22      avg_glucose_level_original   -60.266902             60.266902
    19           smoking_status_smokes    49.196126             49.196126
    4                              bmi    47.990535             47.990535
    17  smoking_status_formerly smoked   -41.602186             41.602186
    1                     hypertension    34.952088             34.952088
    0                              age    33.288465             33.288465
    12               work_type_Private    32.074784             32.074784
    13         work_type_Self-employed   -31.362137             31.362137
    2                    heart_disease    23.068512             23.068512
    3                avg_glucose_level    16.959317             16.959317
    18     smoking_status_never smoked   -15.652458             15.652458
    15            Residence_type_Rural   -11.936047             11.936047
    8                  ever_married_No   -11.317024             11.317024
    5                    gender_Female    -9.841626              9.841626
    10              work_type_Govt_job    -6.230134              6.230134
    16            Residence_type_Urban     3.877528              3.877528
    9                 ever_married_Yes     3.258506              3.258506
    14              work_type_children    -2.188876              2.188876
    6                      gender_Male     1.783107              1.783107
    11          work_type_Never_worked    -0.352156              0.352156
    7                     gender_Other     0.000000              0.000000
    
    Top 3 factors: 
                           Feature  Coefficient  Absolute_Coefficient
    20                age_original   235.422987            235.422987
    21                bmi_original   109.159248            109.159248
    22  avg_glucose_level_original   -60.266902             60.266902
    
    The top 3 factors that lead to stroke are: age_original, bmi_original, avg_glucose_level_original



```python

# Select only relevant features
X = df[['age_original', 'hypertension', 'heart_disease']]
y = df['stroke']

# Train a decision tree classifier
tree_model = DecisionTreeClassifier(max_depth=7, random_state=42)  # Limiting depth for interpretability
tree_model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=['No Stroke', 'Stroke'], filled=True, rounded=True)
plt.title("Decision Tree for Stroke Prediction")
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_32_0.png)
    


# **Visualization**


```python
import matplotlib.pyplot as plt

# Plot Train vs Test Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Train vs Test Loss Over Iterations')
plt.legend()
plt.show()

```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_34_0.png)
    



```python
# Plot Train vs Test Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy Over Iterations')
plt.legend()
plt.show()
```


    
![png](CMSC320_Final_Project_files/CMSC320_Final_Project_35_0.png)
    


# **Insights and Conclusions**
