import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Reading the dataset
df = pd.read_csv("groceries.csv", header=None)

# Converting dataset into list of lists
records = []
for i in range(0, df.shape[0]):
    records.append([str(df.values[i,j]) for j in range(0, df.shape[1])])

# Creating the frequent itemsets by applying Apriori algorithm
frequent_itemsets = apriori(records, min_support=0.005, min_confidence=0.2, min_lift=3, min_length=2)

# Generating rules using Association Rule Mining
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Printing the rules with Support, Confidence and Lift
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
