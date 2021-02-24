import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8])
dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

df.head()
df.index
df.to_numpy()
df2.to_numpy()
df.T
df.sort_index(axis=1, ascending=False)
df.sort_values(by='B')

# Selecting a single column, which yields a Series, equivalent to df.A:
df['A']
# Selecting via [], which slices the rows.
df[0:3]

## Selection by Label
df.loc[dates[0]]
df.loc["2013-01-03"]

df.loc[:, ['A', 'B']]
df.loc['20130102':'20130104', ['A', 'B']]
df.loc[dates[0], 'A']

## Selection by Position
df.iloc[3:6]
df[3:6]

df.iloc[3:5, 0:2], df.loc[ :,["A","B"]]
df[3:5]

df.iloc[1, 1]

## Boolean Indexing
df[df.A > 0]

df[df > 0]

## Using the isin() method for filtering
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2[df2['E'].isin(['two', 'four'])]
