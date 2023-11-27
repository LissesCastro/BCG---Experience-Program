# Feature Engineering

- [Feature Engineering](#feature-engineering)
  - [1. Import packages](#1-import-packages)
  - [2. Load data](#2-load-data)
  - [3. Feature engineering](#3-feature-engineering)
    - [Difference between off-peak prices in December and preceding January](#difference-between-off-peak-prices-in-december-and-preceding-january)
    - [Percentual difference between last month consumption and mean consumption in the last 12 months](#percentual-difference-between-last-month-consumption-and-mean-consumption-in-the-last-12-months)
    - [Percentual difference between last 12 months consumption and forecast consumption](#percentual-difference-between-last-12-months-consumption-and-forecast-consumption)
    - [Participation of off-peak power price on total price](#participation-of-off-peak-power-price-on-total-price)
    - [Customer contract duration](#customer-contract-duration)
- [Model and Evaluation](#model-and-evaluation)
  - [1. Data Preparation](#1-data-preparation)
    - [1.1 Missing values handling](#11-missing-values-handling)
    - [1.2 Dealing with categorical data](#12-dealing-with-categorical-data)
  - [2. Initial model evaluation and dataframe modifications](#2-initial-model-evaluation-and-dataframe-modifications)
    - [2.1. First RFC model fitting](#21-first-rfc-model-fitting)
    - [2.2. Dataframe modification](#22-dataframe-modification)
    - [2.3. Hyperparameter tuning](#23-hyperparameter-tuning)
    - [2.4 Final model tunnel and evaluation](#24-final-model-tunnel-and-evaluation)
    - [2.5 Checking for overfitting](#25-checking-for-overfitting)
    - [2.6 Checking feature importance on the model](#26-checking-feature-importance-on-the-model)
- [3. Conclusion](#3-conclusion)


---



## 1. Import packages


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
```

---
## 2. Load data


```python
df = pd.read_csv('client_data.csv')
df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')
```


```python
df.head(3)
```




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
      <th>channel_sales</th>
      <th>cons_12m</th>
      <th>cons_gas_12m</th>
      <th>cons_last_month</th>
      <th>date_activ</th>
      <th>date_end</th>
      <th>date_modif_prod</th>
      <th>date_renewal</th>
      <th>forecast_cons_12m</th>
      <th>...</th>
      <th>has_gas</th>
      <th>imp_cons</th>
      <th>margin_gross_pow_ele</th>
      <th>margin_net_pow_ele</th>
      <th>nb_prod_act</th>
      <th>net_margin</th>
      <th>num_years_antig</th>
      <th>origin_up</th>
      <th>pow_max</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24011ae4ebbe3035111d65fa7c15bc57</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>0</td>
      <td>54946</td>
      <td>0</td>
      <td>2013-06-15</td>
      <td>2016-06-15</td>
      <td>2015-11-01</td>
      <td>2015-06-23</td>
      <td>0.00</td>
      <td>...</td>
      <td>t</td>
      <td>0.0</td>
      <td>25.44</td>
      <td>25.44</td>
      <td>2</td>
      <td>678.99</td>
      <td>3</td>
      <td>lxidpiddsbxsbosboudacockeimpuepw</td>
      <td>43.648</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d29c2c54acc38ff3c0614d0a653813dd</td>
      <td>MISSING</td>
      <td>4660</td>
      <td>0</td>
      <td>0</td>
      <td>2009-08-21</td>
      <td>2016-08-30</td>
      <td>2009-08-21</td>
      <td>2015-08-31</td>
      <td>189.95</td>
      <td>...</td>
      <td>f</td>
      <td>0.0</td>
      <td>16.38</td>
      <td>16.38</td>
      <td>1</td>
      <td>18.89</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>764c75f661154dac3a6c254cd082ea7d</td>
      <td>foosdfpfkusacimwkcsosbicdxkicaua</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>2010-04-16</td>
      <td>2016-04-16</td>
      <td>2010-04-16</td>
      <td>2015-04-17</td>
      <td>47.96</td>
      <td>...</td>
      <td>f</td>
      <td>0.0</td>
      <td>28.60</td>
      <td>28.60</td>
      <td>1</td>
      <td>6.60</td>
      <td>6</td>
      <td>kamkkxfxxuwbdslkwifmmcsiusiuosws</td>
      <td>13.856</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



---

## 3. Feature engineering

### Difference between off-peak prices in December and preceding January

The code below was provided to calculate the feature described above. I had to re-create this feature and then think about ways to build on this feature to create features with a higher predictive power.


```python
price_df = pd.read_csv('price_data.csv')
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
price_df.head()
```




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
      <th>price_date</th>
      <th>price_off_peak_var</th>
      <th>price_peak_var</th>
      <th>price_mid_peak_var</th>
      <th>price_off_peak_fix</th>
      <th>price_peak_fix</th>
      <th>price_mid_peak_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-01-01</td>
      <td>0.151367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.266931</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-02-01</td>
      <td>0.151367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.266931</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-03-01</td>
      <td>0.151367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.266931</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-04-01</td>
      <td>0.149626</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.266931</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>038af19179925da21a25619c5a24b745</td>
      <td>2015-05-01</td>
      <td>0.149626</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.266931</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group off-peak prices by companies and month
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()
```




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
      <th>offpeak_diff_dec_january_energy</th>
      <th>offpeak_diff_dec_january_power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002203ffbb812588b632b9e628cc38d</td>
      <td>-0.006192</td>
      <td>0.162916</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0004351ebdd665e6ee664792efc4fd13</td>
      <td>-0.004104</td>
      <td>0.177779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0010bcc39e42b3c2131ed2ce55246e3c</td>
      <td>0.050443</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0010ee3855fdea87602a5b7aba8e42de</td>
      <td>-0.010018</td>
      <td>0.162916</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00114d74e963e47177db89bc70108537</td>
      <td>-0.003994</td>
      <td>-0.000001</td>
    </tr>
  </tbody>
</table>
</div>



Inspecionando a feature pré-estabelecida.


```python
#Joining the dataframes
df = df.merge(diff, on='id')
```


```python
#Plotting KDE plots of power offpeak difference and energy offpeak difference between january and december for churned and non churned customers.
fig, ax = plt.subplots(2, 1, figsize=(10,10))
sns.kdeplot(data=df, x = 'offpeak_diff_dec_january_energy', hue='churn', ax=ax[0])
sns.kdeplot(data=df, x = 'offpeak_diff_dec_january_power', hue='churn', ax=ax[1])
```




    <AxesSubplot: xlabel='offpeak_diff_dec_january_power', ylabel='Density'>




    
![png](feature_engineering_files/feature_engineering_10_1.png)
    



```python
#Verifying if the correlation between the churn and the previously created feature are statistically significant
print(pearsonr(df['churn'], df['offpeak_diff_dec_january_power']))
print(pearsonr(df['churn'], df['offpeak_diff_dec_january_energy']))
```

    PearsonRResult(statistic=0.0026715299322503114, pvalue=0.746815315483335)
    PearsonRResult(statistic=-0.0010644314844720326, pvalue=0.8976494793845312)
    

It's not recomended to use the feature provided, since it has high p-value. We couldn't atest that there's a statistically significant correlation between these features and churn, so it may not be efficient in churn prediction. However, since its use is mandatory for the experience program, i will follow with this feature on the prediction model.

### Percentual difference between last month consumption and mean consumption in the last 12 months

A fase de EDA apontou que existe uma grande diferença no consumo ocorrido entre quem fez churn ou não. Assim, achei proveitoso criar uma feature que analisa a diferença de consumo no último mês, em relação à média do período, esperando que o valor do consumo seja menor entre as pessoas que realizaram churn.


```python
#Creating the new feature 
df['consumption_diff_mean_12m_lastmonth_percent'] = df['cons_last_month']-(df['cons_12m']/12)
```


```python
#Plotting KDE plots to inspect the distribution of the feature
fig, ax = plt.subplots(figsize=(10,10))
sns.kdeplot(data=df, x = 'consumption_diff_mean_12m_lastmonth_percent', hue='churn')
```




    <AxesSubplot: xlabel='consumption_diff_mean_12m_lastmonth_percent', ylabel='Density'>




    
![png](feature_engineering_files/feature_engineering_16_1.png)
    


The distribution has tails for both sides, the dinamic of the feature is similar for both the churn groups


```python
#Filling NA's with 0
df['consumption_diff_mean_12m_lastmonth_percent'].fillna(0, inplace=True)
df['consumption_diff_mean_12m_lastmonth_percent'].describe()
```




    count     14606.000000
    mean       2821.912564
    std       21686.353629
    min     -194525.250000
    25%        -757.375000
    50%        -105.125000
    75%         496.312500
    max      449656.000000
    Name: consumption_diff_mean_12m_lastmonth_percent, dtype: float64




```python
#Checking correlation and significance
pearsonr(df['churn'], df['consumption_diff_mean_12m_lastmonth_percent'])
```




    PearsonRResult(statistic=-0.03310565915594221, pvalue=6.287617049842229e-05)



The test indicates that there's a statistically signifcant correlation between the features. It's a negative correlation, which indicates that consuption is lower for the churn group.

### Percentual difference between last 12 months consumption and forecast consumption

Similarly to the feature above, this feature is based on the findings of the EDA phase, atesting that the consumption is lower between churn clients


```python
#Creating the feature
df['diff_12m_forecast'] = df['forecast_cons_12m']/df['cons_12m']

#Since there are 0s on the denomitator, we need to deal with inf numbers
df['diff_12m_forecast'] = df['diff_12m_forecast'].replace([np.inf, -np.inf], 0)
df['diff_12m_forecast'] = df['diff_12m_forecast'].fillna(0)
```


```python
df['diff_12m_forecast'].describe()
```




    count    14606.000000
    mean         0.093119
    std          0.055194
    min          0.000000
    25%          0.042855
    50%          0.102308
    75%          0.147460
    max          0.624622
    Name: diff_12m_forecast, dtype: float64




```python
#Plotting KDE plots to inspect the distribution of the feature
fig, ax = plt.subplots(figsize=(10,5))
sns.kdeplot(data=df, x = 'diff_12m_forecast', hue='churn')
```




    <AxesSubplot: xlabel='diff_12m_forecast', ylabel='Density'>




    
![png](feature_engineering_files/feature_engineering_25_1.png)
    


The ratio of consumption has peaks around 1%, 10% and 15%, these peaks are higher for non-churn clients


```python
#Checking correlation and significance
pearsonr(df['churn'], df['diff_12m_forecast'])
```




    PearsonRResult(statistic=0.01389992955005203, pvalue=0.09299286346148429)



In this case, the statistical significance of the feature is relatively high (9%), and the correlation indicates that consumption within the churn base is higher than in the non-churn group. Since there's a lot of features in the model and the significance of this feature is somewhat low, we will use it.

### Participation of off-peak power price on total price

Given that the Exploratory Data Analysis (EDA) findings suggest that the off-peak price was crucial for predicting churn, this feature highlights the significance of this type of consumption in the overall bill. It is anticipated that clients with higher off-peak participation are more likely to churn.


```python
#Grouping the data by id and price
monthly_price_by_id = price_df.groupby(['id', 'price_date']).mean().reset_index()
```


```python
#Evaluating the weight of offpeak price on total bill price
monthly_price_by_id['percent_off_peak_var'] = (monthly_price_by_id['price_off_peak_var']/(monthly_price_by_id['price_off_peak_var']+monthly_price_by_id['price_peak_var']+monthly_price_by_id['price_mid_peak_var']))*100
monthly_price_by_id['percent_off_peak_fix'] = (monthly_price_by_id['price_off_peak_fix']/(monthly_price_by_id['price_off_peak_fix']+monthly_price_by_id['price_peak_fix']+monthly_price_by_id['price_mid_peak_fix']))*100
```


```python
#Joining the dataframes
df = df.merge(monthly_price_by_id[['id', 'percent_off_peak_var', 'percent_off_peak_fix']], on='id')

#Filling NA's with zeros
df['percent_off_peak_fix'].fillna(0, inplace=True)
df['percent_off_peak_var'].fillna(0, inplace=True)
```


```python
df[['percent_off_peak_var', 'percent_off_peak_fix']].describe()
```




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
      <th>percent_off_peak_var</th>
      <th>percent_off_peak_fix</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>175149.000000</td>
      <td>175149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>72.538129</td>
      <td>80.285580</td>
    </tr>
    <tr>
      <th>std</th>
      <td>27.412829</td>
      <td>25.117158</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41.622809</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66.446392</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Plotting KDE plots to inspect the distribution of the features
fig, ax = plt.subplots(2, 1, figsize=(10,10))
sns.kdeplot(data=df, x = 'percent_off_peak_var', hue='churn', ax=ax[0])
sns.kdeplot(data=df, x = 'percent_off_peak_fix', hue='churn', ax=ax[1])
```




    <AxesSubplot: xlabel='percent_off_peak_fix', ylabel='Density'>




    
![png](feature_engineering_files/feature_engineering_35_1.png)
    



```python
#Checking correlation and significance
print(pearsonr(df['percent_off_peak_var'], df['churn']))
print(pearsonr(df['percent_off_peak_fix'], df['churn']))
```

    PearsonRResult(statistic=-0.033870647332244264, pvalue=1.2340886846445667e-45)
    PearsonRResult(statistic=-0.034764430643611774, pvalue=5.549828262152052e-48)
    

There is a statistically significant correlation between the features; however, the interpretation does not align with what was previously identified in the Exploratory Data Analysis (EDA). We will include this feature in the model because, given the multitude of features influencing the energy price, churn may occur among clients who use less off-peak energy but could be affected by other factors.

### Customer contract duration

From a business perspective, it is reasonable to assume that clients with longer contract durations are less likely to experience churn. This feature examines the contract duration, measured in years, for each client.


```python
#Making the feature 
df['contract_duration'] = df['date_end'] - df['date_activ']
df['contract_duration'] = df['contract_duration'].dt.days
df['contract_duration'] = df['contract_duration']/365
```


```python
df['contract_duration'].describe()
```




    count    175149.000000
    mean          5.500536
    std           1.657254
    min           2.002740
    25%           4.002740
    50%           5.010959
    75%           6.449315
    max          13.136986
    Name: contract_duration, dtype: float64




```python
#Plotting KDE plots to inspect the distribution of the features
fig, ax = plt.subplots(figsize=(10,5))
sns.kdeplot(data=df, x = 'contract_duration', hue='churn')
```




    <AxesSubplot: xlabel='contract_duration', ylabel='Density'>




    
![png](feature_engineering_files/feature_engineering_42_1.png)
    



```python
#Checking correlation and significance
print(pearsonr(df['contract_duration'], df['churn']))
```

    PearsonRResult(statistic=-0.07383703963539254, pvalue=3.115230934310429e-210)
    

As anticipated, there is a correlation between contract duration and churn. Clients with longer contract periods are less likely to experience churn.

# Model and Evaluation


```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, LearningCurveDisplay, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
```

We gonna use RandomForest algorithm to perform a classification model on our data and try to predict the churn. Since RandomForest works pretty fine with outlier data, few modifications on dataframe are necessary, related to null values and categorical data. We will first encode our categorical data, so it don't throw any error when performing the algorithm; then we will split our model and train it. After the first evaluation, we will cross-validate the model and check feature importance. 

## 1. Data Preparation


```python
#Train_test split
target = 'churn'

X_train, X_test, y_train, y_test = train_test_split(df.drop([target, 'id', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal'], axis=1), df[target], test_size=0.2, random_state=42)
```

### 1.1 Missing values handling


```python
#Counting the number of missing values in each column
df.isna().sum()
```




    id                                             0
    channel_sales                                  0
    cons_12m                                       0
    cons_gas_12m                                   0
    cons_last_month                                0
    date_activ                                     0
    date_end                                       0
    date_modif_prod                                0
    date_renewal                                   0
    forecast_cons_12m                              0
    forecast_cons_year                             0
    forecast_discount_energy                       0
    forecast_meter_rent_12m                        0
    forecast_price_energy_off_peak                 0
    forecast_price_energy_peak                     0
    forecast_price_pow_off_peak                    0
    has_gas                                        0
    imp_cons                                       0
    margin_gross_pow_ele                           0
    margin_net_pow_ele                             0
    nb_prod_act                                    0
    net_margin                                     0
    num_years_antig                                0
    origin_up                                      0
    pow_max                                        0
    churn                                          0
    offpeak_diff_dec_january_energy                0
    offpeak_diff_dec_january_power                 0
    consumption_diff_mean_12m_lastmonth_percent    0
    diff_12m_forecast                              0
    percent_off_peak_var                           0
    percent_off_peak_fix                           0
    contract_duration                              0
    dtype: int64



Since there's no more null data, it's not necessary to fill it at this point

### 1.2 Dealing with categorical data


```python
#Making a list of categorical data from the variable
cat_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']
cat_columns
```




    ['channel_sales', 'has_gas', 'origin_up']




```python
#Using Label Encoder to encode categorical data
le = LabelEncoder()
for col in cat_columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
```

## 2. Initial model evaluation and dataframe modifications

### 2.1. First RFC model fitting

First, we gonna train a model with some set parameters and check it's accuracy, then we can see the most important features and hyperparameter tunning. 


```python
rfc = RandomForestClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2)
model = rfc.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
accuracy_score(y_test, y_pred)
```




    0.9155866400228375



The first model got 91,55% accuracy. 


```python
confusion_matrix(y_test, y_pred)
```




    array([[31705,     0],
           [ 2957,   368]], dtype=int64)



Checking feature importance to understand what are the most important features in the model and remove soma useless features. RandomForestClassifier performance should not be worsened by these features, since its handle very well if this number of variables, but it's a good practice to remove unused features from the model.


```python
feature_importances = model.feature_importances_
# Associate feature names with their importance scores
feature_names = X_train.columns
feature_importance_dict = dict(zip(feature_names, feature_importances))
# Sort features by importance
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(27).plot(kind='barh')
```




    <AxesSubplot: >




    
![png](feature_engineering_files/feature_engineering_64_1.png)
    


### 2.2. Dataframe modification

The initial model achieved approximately 91.55% accuracy, which is a commendable result. However, there is room for improvement, as an error rate of 9% in certain cases is suboptimal. To enhance the model's performance, we will begin by eliminating certain features. Specifically, we plan to remove features identified as having low impact on churn through Exploratory Data Analysis (EDA). These features include the variable indicating whether the client uses gas, the number of products held by the client, and the channel through which the client received the offer. Additionally, we will exclude any created features that exhibit the lowest importance according to our analysis.


```python
del_features = ['percent_off_peak_fix', 'has_gas', 'percent_off_peak_var', 'nb_prod_act', 'channel_sales']

for col in del_features:
    X_train.drop(col, axis=1, inplace=True)
    X_test.drop(col, axis=1, inplace=True)
```

### 2.3. Hyperparameter tuning

We will set a range where we RandomSearchCV will work on. It will perform 50 random parameter settings, applying it at 5 folds a time. This work will generate a set of parameters that are almost the best possible.


```python
#Setting some limits for the random_grid to work on and perform a pre-selection of the best parameters.

n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=100)]
max_depth = [int(x) for x in np.linspace(1, 100, 10)]
min_samples_leaf = [5, 10, 20, 40, 100]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
                }
```


```python
modeL_RSCV = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=50, cv=5, random_state=42) #n_jobs=-1)
modeL_RSCV.fit(X_train, y_train) 
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,
                   estimator=RandomForestClassifier(max_depth=10,
                                                    min_samples_leaf=2,
                                                    min_samples_split=5,
                                                    random_state=42),
                   n_iter=50,
                   param_distributions={&#x27;max_depth&#x27;: [1, 12, 23, 34, 45, 56, 67,
                                                      78, 89, 100],
                                        &#x27;min_samples_leaf&#x27;: [5, 10, 20, 40,
                                                             100],
                                        &#x27;n_estimators&#x27;: [100, 109, 118, 127,
                                                         136, 145, 154, 163,
                                                         172, 181, 190, 200,
                                                         209, 218, 227, 236,
                                                         245, 254, 263, 272,
                                                         281, 290, 300, 309,
                                                         318, 327, 336, 345,
                                                         354, 363, ...]},
                   random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=5,
                   estimator=RandomForestClassifier(max_depth=10,
                                                    min_samples_leaf=2,
                                                    min_samples_split=5,
                                                    random_state=42),
                   n_iter=50,
                   param_distributions={&#x27;max_depth&#x27;: [1, 12, 23, 34, 45, 56, 67,
                                                      78, 89, 100],
                                        &#x27;min_samples_leaf&#x27;: [5, 10, 20, 40,
                                                             100],
                                        &#x27;n_estimators&#x27;: [100, 109, 118, 127,
                                                         136, 145, 154, 163,
                                                         172, 181, 190, 200,
                                                         209, 218, 227, 236,
                                                         245, 254, 263, 272,
                                                         281, 290, 300, 309,
                                                         318, 327, 336, 345,
                                                         354, 363, ...]},
                   random_state=42)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5,
                       random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5,
                       random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>



Getting the best parameters for the model


```python
modeL_RSCV.best_params_
```




    {'n_estimators': 936, 'min_samples_leaf': 5, 'max_depth': 67}



### 2.4 Final model tunnel and evaluation


```python
#Fitting a new RandomForestClassifier model, utilizing the best parameters found
rfc = RandomForestClassifier(random_state=42, n_estimators= 936, min_samples_leaf= 5, max_depth= 67)
model = rfc.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


```python
#Getting performance metrics for the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
```

    Accuracy: 0.9993434199257779
    Precision: 1.0
    Recall: 0.9930827067669173
    

The final model shows good overall performance. It has an accuracy of 99% to detect churning clients. Still, it's interesting to cross validate this model and check if it has overfitting.

### 2.5 Checking for overfitting

We gonna perform cross validation to check wheter there's overfitting on our sample. Cross validation performs model on a set of 10 folds and verify the mean accuracy scores among these folds.


```python
#Performing cross validation to check score, using 10 folds.
scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
```


```python
#Check mean accuracy score among 10 folds
scores.mean()
```




    0.9996146145152907



the cross validation evaluation shows that there's 99% accuracy on the model across 10 folds, which suggests that there's no overfitting within the sample.

### 2.6 Checking feature importance on the model


```python
feature_importances = model.feature_importances_

feature_names = X_train.columns
feature_importance_dict = dict(zip(feature_names, feature_importances))
# Sort features by importance
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
```




    <AxesSubplot: >




    
![png](feature_engineering_files/feature_engineering_84_1.png)
    


# 3. Conclusion

- The final model has a 99% accuracy rate in predicting churning clients.
- PowerCO's price sensitivity hypothesis seems to be inaccurate since the majority of price features don't show a high correlation with churn and appear to be important in detecting churn prevention.
- Consumption features are more definitive in predicting client churn. This aligns with EDA findings, which show a significant difference in consumption between churning and non-churning clients. Last month's consumption is a practical feature to check for churning clients.
- Some created features had high predictive power. Examples include the "Difference in consumption between last month and the mean of the last 12 months," "Difference between forecast consumption and real consumption for 12 months," and "Off-peak power price difference between December and January."
