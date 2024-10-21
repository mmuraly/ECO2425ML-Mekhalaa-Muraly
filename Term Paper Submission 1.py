#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Rename variables based on their labels
rename_dict = {
    'ssuid': 'Sample_Unit_Identifier',
    'spanel': 'Sample_Code_Panel_Year',
    'swave': 'Wave_of_Data_Collection',
    'srefmon': 'Reference_Month',
    'rhcalmn': 'Calendar_Month',
    'rhcalyr': 'Calendar_Year',
    'tfipsst': 'FIPS_State_Code',
    'epppnum': 'Person_Number',
    'esex': 'Sex',
    'wpfinwgt': 'Person_Weight',
    'tage': 'Age',
    'eeducate': 'Education_Level',
    'rmesr': 'Labour_Market_Participation',
    'birth_month': 'Birth_Month_Year'
}
df.rename(columns=rename_dict, inplace=True)

# Display general information about the dataset
df.info()

# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns:\n")
print(df.describe())


# In[8]:


# The initial simple difference-in-difference code is obtained from the original paper Byker (2016)
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
# 'ssuid' should be a string of length 12
df['ssuid'] = df['ssuid'].astype(str)

# Ensure integer types for relevant columns
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')

# Double type column
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)

# Create a unique ID for each individual
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable (as float)
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to a Period type for monthly representation
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')

# Create 'date' variable using year and month and convert to %tm format
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names and make it categorical
state_labels = {6: "California", 34: "New Jersey", 12: "Florida", 48: "Texas", 36: "New York"}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Find the last date for each individual and format '%tm'
df['end_date'] = df.groupby('sippid')['date'].transform('max')

# Get the weight from the last observation
df['end_weight_f'] = df.apply(lambda x: x['wpfinwgt'] if x['date'] == x['end_date'] else np.nan, axis=1).astype(float)
df['end_weight'] = df.groupby('sippid')['end_weight_f'].transform('max').astype(float)

# Round the weights
df['end_weight'] = df['end_weight'].round()

# Define policy implementation dates as period '%tm'
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')

# Generate variable indicating births that occurred when a paid leave was in effect in the mother's state
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable (treated as categorical for difference-in-difference model)
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = ((df['state'].isin(['California', 'New Jersey'])) & (df['post_policy'] == 1)).astype(int)

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Difference-in-Difference regression model
formula = 'rm_lfp ~ treated + C(state) + post_policy + treated:post_policy'
model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['sippid']})

# Print the summary of the regression model
print(model.summary())


# In[26]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {6: "California", 34: "New Jersey", 12: "Florida", 48: "Texas", 36: "New York"}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Find the last date for each individual and get the weight from the last observation
df['end_date'] = df.groupby('sippid')['date'].transform('max')
df['end_weight_f'] = df.apply(lambda x: x['wpfinwgt'] if x['date'] == x['end_date'] else np.nan, axis=1).astype(float)
df['end_weight'] = df.groupby('sippid')['end_weight_f'].transform('max').astype(float)
df['end_weight'] = df['end_weight'].round()

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = ((df['state'].isin(['California', 'New Jersey'])) & (df['post_policy'] == 1)).astype(int)

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Run Logit regression model for Difference-in-Difference without clustered standard errors
formula = 'rm_lfp ~ treated:post_policy + C(state)'
try:
    model = smf.logit(formula, data=df).fit()
    # Print the summary of the regression model
    print(model.summary())
except np.linalg.LinAlgError:
    # If the Hessian is singular, run without clustering
    model = smf.logit(formula, data=df).fit(method='bfgs')
    print(model.summary())


# In[17]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {6: "California", 34: "New Jersey", 12: "Florida", 48: "Texas", 36: "New York"}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Find the last date for each individual and get the weight from the last observation
df['end_date'] = df.groupby('sippid')['date'].transform('max')
df['end_weight_f'] = df.apply(lambda x: x['wpfinwgt'] if x['date'] == x['end_date'] else np.nan, axis=1).astype(float)
df['end_weight'] = df.groupby('sippid')['end_weight_f'].transform('max').astype(float)
df['end_weight'] = df['end_weight'].round()

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables (keep only treated_post_policy to reduce multicollinearity)
df['treated_post_policy'] = ((df['state'].isin(['California', 'New Jersey'])) & (df['post_policy'] == 1)).astype(int)

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated_post_policy', 'state'])

# Prepare features and target variable for modeling
df = pd.get_dummies(df, columns=['state'], drop_first=True)
X = df[['treated_post_policy'] + [col for col in df.columns if col.startswith('state_')]]
y = df['rm_lfp']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso path using lasso_path to obtain coefficients for different values of alpha
alphas, coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-10, 1, 100))

# Plot the Lasso coefficient paths
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(coefs.shape[0]):
    ax.plot(-np.log(alphas), coefs[i], label=X.columns[i])

ax.legend(loc='upper left')
ax.set_xlabel('$-\\log(\\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
plt.show()

# Cross-validated accuracy plot for logistic regression with L1 regularization
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logisticCV = LogisticRegressionCV(
    Cs=10,  # Number of regularization strengths
    cv=kfold,
    penalty='l1',  # Lasso-like penalty
    solver='saga',  # 'saga' supports L1 penalty
    random_state=42,
    max_iter=500
)

# Fit the model
pipeCV = Pipeline(steps=[('scaler', scaler), ('logistic', logisticCV)])
pipeCV.fit(X, y)

# Output tuned alpha (C) and minimum cross-validated MSE path mean
tuned_logistic = pipeCV.named_steps['logistic']
print("Tuned Alpha (Regularization Strength) for Lasso: ", tuned_logistic.C_[0])

# Cross-validated accuracy plot
mean_scores = np.mean(tuned_logistic.scores_[1], axis=0)  # Use scores for class "1" (assuming binary target)
std_scores = np.std(tuned_logistic.scores_[1], axis=0)

fig, ax = plt.subplots(figsize=(8, 8))
ax.errorbar(
    -np.log(tuned_logistic.Cs_),
    mean_scores,
    yerr=std_scores / np.sqrt(kfold.get_n_splits()),
    fmt='o',
    ecolor='lightgray',
    elinewidth=2,
    capsize=4,
    color='blue'
)
ax.axvline(-np.log(tuned_logistic.C_[0]), color='k', linestyle='--')
ax.set_xlabel('$-\\log(\\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated Accuracy', fontsize=20)
plt.show()


# In[21]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {6: "California", 34: "New Jersey", 12: "Florida", 48: "Texas", 36: "New York"}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Find the last date for each individual and get the weight from the last observation
df['end_date'] = df.groupby('sippid')['date'].transform('max')
df['end_weight_f'] = df.apply(lambda x: x['wpfinwgt'] if x['date'] == x['end_date'] else np.nan, axis=1).astype(float)
df['end_weight'] = df.groupby('sippid')['end_weight_f'].transform('max').astype(float)
df['end_weight'] = df['end_weight'].round()

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create treated vs control group variable
df['treated'] = df['state'].apply(lambda x: 1 if x in ["California", "New Jersey"] else 0)

# Interaction term for difference-in-difference
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Create a control group indicator (non-treated states)
df['control'] = 1 - df['treated']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated_post_policy', 'treated', 'control', 'state'])

# Prepare features and target variable for modeling
X = df[['treated_post_policy', 'treated', 'control']]
y = df['rm_lfp']

# Train a Random Forest model to estimate the impact
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar([X.columns[i] for i in indices], importances[indices])
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# In[24]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {
    6: "California",
    34: "New Jersey",
    12: "Florida",
    48: "Texas",
    36: "New York"
}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['state'].isin(['California', 'New Jersey']).astype(int)
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Prepare features and target variable for modeling
df = pd.get_dummies(df, columns=['state'], drop_first=False)
X = df[['treated', 'treated_post_policy'] + [col for col in df.columns if col.startswith('state_')]]
y = df['rm_lfp']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
importances = rf.feature_importances_
feature_names = ['treated', 'treated_post_policy'] + [col.split('_', 1)[-1] for col in df.columns if col.startswith('state_')]

# Plot the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(importances)), importances, tick_label=feature_names)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Feature Importances', fontsize=15)
plt.tight_layout()
plt.show()


# In[26]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {
    6: "California",
    34: "New Jersey",
    12: "Florida",
    48: "Texas",
    36: "New York"
}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['state'].isin(['California', 'New Jersey']).astype(int)
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Prepare features and target variable for modeling
df = pd.get_dummies(df, columns=['state'], drop_first=False)
X = df[['treated', 'treated_post_policy'] + [col for col in df.columns if col.startswith('state_')]]
y = df['rm_lfp']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_hat_RF = rf.predict(X_test)

# Calculate Mean Squared Error
rf_mse = mean_squared_error(y_test, y_hat_RF)
print(f'Random Forest Model Mean Squared Error: {rf_mse}')

# Get feature importance
importances = rf.feature_importances_
feature_names = ['treated', 'treated_post_policy'] + [col.split('_', 1)[-1] for col in df.columns if col.startswith('state_')]

# Plot the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(importances)), importances, tick_label=feature_names)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Feature Importances', fontsize=15)
plt.tight_layout()
plt.show()


# In[27]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {
    6: "California",
    34: "New Jersey",
    12: "Florida",
    48: "Texas",
    36: "New York"
}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['state'].isin(['California', 'New Jersey']).astype(int)
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Prepare features and target variable for modeling
df = pd.get_dummies(df, columns=['state'], drop_first=False)
X = df[['treated', 'treated_post_policy'] + [col for col in df.columns if col.startswith('state_')]]
y = df['rm_lfp']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=0)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)
print("Accuracy Score (Test Set):", accuracy_score(y_test, clf.predict(X_test)))

# Residual Deviance (Log Loss)
resid_dev = np.sum(log_loss(y_train, clf.predict_proba(X_train)))
print("Residual Deviance:", resid_dev)

# Plot the Decision Tree
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf, feature_names=X.columns, filled=True, ax=ax)
plt.show()

# Text representation of the tree
print(export_text(clf, feature_names=list(X.columns), show_weights=True))

# Cross-validation using ShuffleSplit
validation = ShuffleSplit(n_splits=1, test_size=200, random_state=0)
results = cross_validate(clf, X_scaled, y, cv=validation)
print("Test Score (Cross-Validation):", results['test_score'])

# Pruning the tree using cost complexity pruning
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
kfold = KFold(10, random_state=1, shuffle=True)
grid = GridSearchCV(clf, {'ccp_alpha': ccp_path.ccp_alphas}, refit=True, cv=kfold, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best Score after Pruning:", grid.best_score_)

# Plotting the best tree after pruning
best_ = grid.best_estimator_
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(best_, feature_names=X.columns, filled=True, ax=ax)
plt.show()

# Number of leaves in the best pruned tree
print("Number of Leaves in Best Tree:", best_.tree_.n_leaves)

# Accuracy of the pruned tree on the test set
print("Accuracy Score (Pruned Tree, Test Set):", accuracy_score(y_test, best_.predict(X_test)))

# Confusion Matrix
confusion = confusion_matrix(y_test, best_.predict(X_test))
print("Confusion Matrix (Test Set):")
print(confusion)


# In[10]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {
    6: "California",
    34: "New Jersey",
    12: "Florida",
    48: "Texas",
    36: "New York"
}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['state'].isin(['California', 'New Jersey']).astype(int)
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Prepare features and target variable for modeling
df = pd.get_dummies(df, columns=['state'], drop_first=False)
X = df[['treated', 'treated_post_policy'] + [col for col in df.columns if col.startswith('state_')]]
y = df['rm_lfp']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Boosting Model
boosting_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,  # Lower learning rate for better control
    max_depth=3,
    min_samples_split=10,  # Helps control overfitting
    subsample=0.8,  # Adds regularization
    random_state=42
)
boosting_model.fit(X_train, y_train)

# Predictions and Error Calculations
y_hat_boost = boosting_model.predict(X_test)

# Mean Squared Error for Boosting Model
mse_boost = mean_squared_error(y_test, y_hat_boost)

# Accuracy for Boosting Model
accuracy_boost = accuracy_score(y_test, y_hat_boost)

print(f"Boosting Model Mean Squared Error: {mse_boost}")
print(f"Boosting Model Accuracy: {accuracy_boost}")


# In[13]:


import pandas as pd
import numpy as np
get_ipython().system('pip install dowhy')
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Generate 'months' variable
df['months'] = df.groupby('sippid').cumcount() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Generate the 'birth' variable that indicates month relative to birth
df['birth'] = (df['date'] - df['birth_month']).apply(lambda x: x.n if not pd.isna(x) else np.nan)
df['birth'] = df['birth'].astype(float)

# Handle missing birth observations
df['birth_seen_f'] = (df['birth'] == 0).astype(int)
df['birth_seen'] = df.groupby('sippid')['birth_seen_f'].transform('max').astype(int)

# Find the earliest 'birth' value for each individual where birth > 0 and birth not seen
def min_birth(x):
    cond = (x['birth'] > 0) & (x['birth_seen'] == 0)
    return x.loc[cond, 'birth'].min() if cond.any() else np.nan

df['ref_month_ns'] = df.groupby('sippid').apply(min_birth).reset_index(level=0, drop=True).astype(float)
df['ref_month'] = np.nan
df.loc[(df['birth'] == 0) & (df['birth_seen'] == 1), 'ref_month'] = 1
df.loc[(df['ref_month_ns'] == df['birth']) & (df['birth_seen'] == 0), 'ref_month'] = 1
df['ref_month'] = df['ref_month'].astype(float)

# Map state codes to state names
state_labels = {
    6: "California",
    34: "New Jersey",
    12: "Florida",
    48: "Texas",
    36: "New York"
}
df['state'] = df['tfipsst'].map(state_labels)
df['state'] = df['state'].astype('category')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['state'].isin(['California', 'New Jersey']).astype(int)
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'state', 'post_policy'])

# Define DAG structure using gml_graph for causal analysis
gml_graph = """
graph [
    directed 1
    node [
        id 0
        label "treated_post_policy"
    ]
    node [
        id 1
        label "rm_lfp"
    ]
    node [
        id 2
        label "tage"
    ]
    node [
        id 3
        label "eeducate"
    ]
    node [
        id 4
        label "esex"
    ]
    edge [
        source 0
        target 1
    ]
    edge [
        source 2
        target 1
    ]
    edge [
        source 3
        target 1
    ]
    edge [
        source 4
        target 1
    ]
    edge [
        source 2
        target 0
    ]
    edge [
        source 3
        target 0
    ]
    edge [
        source 4
        target 0
    ]
]
"""

# Create causal model using dowhy
model = CausalModel(
    data=df,
    treatment='treated_post_policy',
    outcome='rm_lfp',
    graph=gml_graph
)

# View the causal model graph
model.view_model()

# Perform identification, estimation, and refutation
identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
print(estimate)

# Refutation
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
print(refute_results)


# In[4]:


import pandas as pd
import numpy as np
from dowhy import CausalModel
import pygraphviz as pgv
import matplotlib.pyplot as plt

# Load the data
df = pd.read_stata('SIPP_Paid_Leave.dta')

# Convert data types to match the original dataset
df['ssuid'] = df['ssuid'].astype(str)
df['spanel'] = df['spanel'].astype(int)
df['epppnum'] = df['epppnum'].astype(int)
df['tfipsst'] = df['tfipsst'].astype('int8')
df['rhcalyr'] = df['rhcalyr'].astype(int)
df['swave'] = df['swave'].astype('int8')
df['srefmon'] = df['srefmon'].astype('int8')
df['rhcalmn'] = df['rhcalmn'].astype('int8')
df['tage'] = df['tage'].astype('int8')
df['eeducate'] = df['eeducate'].astype('int8')
df['rmesr'] = df['rmesr'].astype('int8')
df['esex'] = df['esex'].astype('int8')
df['wpfinwgt'] = df['wpfinwgt'].astype(float)

# Sort the data and create unique ID for each individual
df.sort_values(by=['ssuid', 'epppnum', 'spanel', 'swave', 'srefmon'], inplace=True)
df['sippid'] = df.groupby(['spanel', 'ssuid', 'epppnum']).ngroup() + 1

# Convert 'birth_month' to datetime and create 'date' variable
df['birth_month'] = pd.to_datetime(df['birth_month'], format='%Y-%m').dt.to_period('M')
df['date'] = pd.to_datetime(dict(year=df['rhcalyr'], month=df['rhcalmn'], day=1)).dt.to_period('M')

# Define policy implementation dates and generate 'post_policy' variable
df['CA_date'] = pd.Period('2004-07', freq='M')
df['NJ_date'] = pd.Period('2009-07', freq='M')
df['post_policy'] = 0
df.loc[(df['tfipsst'] == 34) & (df['birth_month'] >= df['NJ_date']), 'post_policy'] = 1
df.loc[(df['tfipsst'] == 6) & (df['birth_month'] >= df['CA_date']), 'post_policy'] = 1

# Labor force participation variable
df['rm_lfp'] = (df['rmesr'] <= 7).astype(int)
df.loc[df['rmesr'] == -1, 'rm_lfp'] = np.nan

# Create difference-in-difference variables
df['treated'] = df['tfipsst'].isin([6, 34]).astype(int)  # California (6) and New Jersey (34) are treated states
df['treated_post_policy'] = df['treated'] * df['post_policy']

# Drop rows with missing values to avoid issues in regression
df = df.dropna(subset=['rm_lfp', 'treated', 'post_policy', 'tage', 'eeducate', 'esex'])

# Defining controls
controls = ['esex', 'tage', 'eeducate']

# Create a GML representation of the DAG
gml_graph = """
graph[directed 1
      node[ id "treated_post_policy" label "treated_PostPolicy" ]
      node[ id "rm_lfp" label "Labour Force Participation" ]
      node[ id "esex" label "Sex" ]
      node[ id "tage" label "Age" ]
      node[ id "eeducate" label "Education" ]
      
      edge[ source "esex" target "rm_lfp" ]
      edge[ source "tage" target "rm_lfp" ]
      edge[ source "eeducate" target "rm_lfp" ]
      edge[ source "esex" target "treated_post_policy" ]
      edge[ source "tage" target "treated_post_policy" ]
      edge[ source "eeducate" target "treated_post_policy" ]
      edge[ source "treated_post_policy" target "rm_lfp" ]
]
"""

# Create causal model using DoWhy
model = CausalModel(
    data=df,
    treatment='treated_post_policy',
    outcome='rm_lfp',
    graph=gml_graph
)

# View the causal graph
model.view_model(layout="dot")
plt.show()

# Identify the causal effect using the backdoor criterion
identified_estimand = model.identify_effect()

# Estimate the causal effect using a linear regression estimator suitable for continuous outcomes
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    control_value=0,
    treatment_value=1
)

# Print the identified estimand and the estimate
print("Identified Estimand:", identified_estimand)
print("Causal Estimate:", causal_estimate)

