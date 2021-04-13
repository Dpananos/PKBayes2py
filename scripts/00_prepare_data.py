import pandas as pd
# Prepare some of the data to be modeled.

df = pd.read_csv('data/rommels_apixiban_data.csv')

new_cols = dict(
time = df.Time,
subjectids = pd.Categorical(df.Subject).codes + 1,
yobs = df.Concentration/1000,
sex = df.Sex.apply(lambda x: 1 if x=='Male' else 0),
age = df.Age,
weight = df.Weight,
creatinine = df.Creatinine,
D = 2.5
)

new_df = pd.DataFrame(new_cols)

new_df.to_csv("data/generated_data/experiment.csv", index = False)





