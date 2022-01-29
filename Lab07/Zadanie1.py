import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('OnlineRetail.csv', encoding= 'unicode_escape')
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]
sns.countplot(x = 'Description', data=data, order=data['Description'].value_counts().iloc[:10].index)
# plt.xticks(rotation=90)
basket = (data[data['Country'] =="Belgium"]
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum().unstack().reset_index().fillna(0)
    .set_index('InvoiceNo'))
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# pl = pd.DataFrame(data, columns=["Description", "Quantity"])
# pl.plot(x="Description", y="Quantity", kind="bar", figsize=(9, 8))
# plt.show()
basket_sets = basket.applymap(encode_units)
basket_df = pd.DataFrame(basket_sets)
freq_items = apriori(basket_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="lift", min_threshold=2)
rules.sort_values('confidence', ascending = False, inplace = True)
rules[ (rules['confidence'] >= 0.7) ]
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
