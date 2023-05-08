import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys

for option in ["single"] :
    for dataset in ["sift1M", "gist1M", "crawl", "msong", "glove-100", "deep1M"] :
        for top_k in [1, 10] :
            f = open(f"./result/{option}/{option}_{dataset}_top{top_k}.txt", 'r')             
            df = pd.DataFrame(columns=["recall", "latency"])
            count = 0             

            while True:
                line = f.readline()
                if not line :
                    break
                    
                if "Query with tau_query" in line :
                    row = [0, 0]
                    count = 0

                elif "[GPU: 0]" in line :
                    count = count + 1
                    m = re.search('=> ms: (.+?) \[', line)
                    if m :
                        latency = float(m.group(1)) # Latency in ms
                        latency = latency * 1000
                        row[1] = row[1] + latency

                elif "r@" in line :
                    m = re.search(' : (.+?) out of', line)
                    if m :
                        correct = float(m.group(1))

                    m = re.search(' out of (.+?)\n', line)
                    if m :
                        total = float(m.group(1))

                    row[0] = correct / total
                    row[1] = row[1] / count
                    df.loc[len(df)] = row
    
            df = df.sort_values(by=["recall", "latency"], ascending=[True, True])
            df = df.drop_duplicates(subset=["recall"], keep="first").T
            df.to_csv(f"./result/ggnn_{option}_batch/{option}_{dataset}_top{top_k}.csv", index=False, header=False)
            f.close()

for option in ["whole"] :
    for dataset in ["sift1M", "gist1M", "crawl", "msong", "glove-100", "deep1M"] :
        for top_k in [1, 10] :
            f = open(f"./result/{option}/{option}_{dataset}_top{top_k}.txt", 'r')
            df = pd.DataFrame(columns=["recall", "QPS"])

            while True:
                line = f.readline()
                if not line :
                    break

                if "Query with tau_query" in line :
                    row = [0, 0]

                elif "[GPU: 0]" in line :
                    m = re.search('=> ms: (.+?) \[', line)
                    if m :
                        latency = float(m.group(1)) # Latency in ms
                        row[1] = latency / 1000 # Latency in s

                elif "r@" in line :
                    m = re.search(' : (.+?) out of', line)
                    if m :
                        correct = float(m.group(1))
                        
                    m = re.search(' out of (.+?)\n', line)
                    if m :
                        total = float(m.group(1))

                    row[0] = correct / total
                    row[1] = (total / top_k) / row[1]
                    df.loc[len(df)] = row

            df = df.sort_values(by=["recall", "QPS"], ascending=[True, False])
            df = df.drop_duplicates(subset=["recall"], keep="first").T
            df.to_csv(f"./result/ggnn_{option}_batch/{option}_{dataset}_top{top_k}.csv", index=False, header=False)
            f.close()

