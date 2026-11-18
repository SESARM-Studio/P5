import pandas as pd
import pyterrier_alpha as pta

old = {
  "docno": ["1","32","2","45","8","3"]
}

new = {
  "docno": ["32","1","2","45","3","8"]
}

df_old = pd.DataFrame(old)
df_new = pd.DataFrame(new)

df_old["qid"] = "q1"
df_old["score"] = range(len(df_old), 0, -1)

df_new["qid"] = "q1"
df_new["score"] = range(len(df_new), 0, -1)

result = pta.rbo(df_old, df_new, p=0.9)


print(list(result))