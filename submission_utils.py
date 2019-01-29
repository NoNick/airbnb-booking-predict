import pandas as pd
import numpy as np

classes = ['AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'NDF', 'NL', 'PT', 'US', 'other']


def top5(p):
    labelsRel = pd.DataFrame(np.argsort(p), columns=['rel'])
    labelsRel['label'] = classes
    return labelsRel.sort_values('rel')['label'][0:5].tolist()


def getTops(probMatrix):
    resultCountries = []
    for prob in probMatrix:
        resultCountries += top5(prob)
    return resultCountries


def saveResult(ids, probMatrix, path):
    f = open(path, "w+")
    f.write("id,country\r\n")
    countries = getTops(probMatrix)
    ids = np.repeat(ids, 5).tolist()
    for user_id, country in zip(ids, countries):
        f.write(user_id + "," + country + "\r\n")
    f.close()
