import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
airline = pd.read_csv('Airline revenue predictor.csv')
airline['Revenue'] = airline['Revenue'].str.replace('$', '').str.replace('B', '').astype(float)
airline['Revenue'] *=1000000000
pd.options.display.float_format = '{:.1f}'.format
Delta = airline[airline['Company'] == 'Delta Airlines']
Singapore = airline[airline['Company'] == 'Singapore Airlines']
Lufthansa = airline[airline['Company'] == 'Lufthansa']
Turkish = airline[airline['Company'] == 'Turkish Airlines']
model_delta = LinearRegression()
delta_x = Delta[['Year']]
delta_y = Delta['Revenue']
model_delta.fit(delta_x, delta_y)
delta_2024 = model_delta.predict([[2024]])
delta_2024_1 = pd.DataFrame({'Year': [2024], 'Revenue': delta_2024, 'Company': 'Delta Airlines'})
final_delta = pd.concat([Delta, delta_2024_1], ignore_index= True).sort_values(by = 'Year', ascending=False)
model_singa = LinearRegression()
singa_x = Singapore[['Year']]
singa_y = Singapore['Revenue']
model_singa.fit(singa_x, singa_y)
singa_2024 = model_singa.predict([[2024]])
singa_2024_1 = pd.DataFrame({'Year': [2024], 'Revenue': singa_2024, 'Company': 'Singapore Airlines'})
final_singa = pd.concat([Singapore, singa_2024_1], ignore_index= True).sort_values(by = 'Year', ascending=False)
model_luft = LinearRegression()
luft_x = Lufthansa[['Year']]
luft_y = Lufthansa['Revenue']
model_luft.fit(luft_x, luft_y)
luft_2024 = model_luft.predict([[2024]])
luft_2024_1 = pd.DataFrame({'Year': [2024], 'Revenue': luft_2024, 'Company': 'Lufthansa'})
final_luft = pd.concat([Lufthansa, luft_2024_1], ignore_index= True).sort_values(by = 'Year', ascending=False)
model_turk = LinearRegression()
turk_x = Turkish[['Year']]
turk_y = Turkish['Revenue']
model_turk.fit(turk_x, turk_y)
turk_2024 = model_turk.predict([[2024]])
turk_2024_1 = pd.DataFrame({'Year': [2024], 'Revenue': turk_2024, 'Company': 'Turkish Airlines'})
final_turk = pd.concat([Turkish, turk_2024_1], ignore_index= True).sort_values(by = 'Year', ascending=False)
final_airline = pd.concat([final_delta, final_singa, final_luft, final_turk], ignore_index=True)
df_pivot = final_airline.pivot(index='Year', columns='Company', values='Revenue')

# Plot the data
ax = df_pivot.plot(kind='line', figsize=(12, 6))
plt.title('Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Revenue (in billions)')
plt.legend(title='Companies', loc='upper left')
plt.grid(True)
plt.xticks(final_airline['Year'].unique())
plt.show()