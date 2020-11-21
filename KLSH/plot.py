import pickle
import pandas as pd
import os

if __name__ == "__main__":
    files = os.listdir("hashes")
    precs = {}
    for file in files:
        if "PrecisionResults" in file:
            with open("hashes/"+ file, 'rb') as f:
                prec = pickle.load(f)
                precs[file[len("PrecisionResults-"):]] = prec[:,0]
    df = pd.DataFrame.from_dict(precs)
    ax = df.plot()
    ax.set_xlabel("# of top images retrieved")
    ax.set_ylabel("Average precision")
    input()