import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = pd.Series([['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
                     ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
                     ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
                     ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
                     ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']])


def apriori_association_mining(dataset, min_support, metric="confidence", min_threshold=0.7):
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    result = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return (frequent_itemsets, result)


(frequent_itemsets, result) = apriori_association_mining(dataset, 0.6)

print((frequent_itemsets, result))
