from sklearn.datasets import load_breast_cancer
import pymongo

# Load the data in dictionary form
dct = load_breast_cancer()

# Taking only required data in dictionary
dct2 = dict()
dct2['data']=dct['data'].tolist()
dct2['target']=dct['target'].tolist()
dct2['feature_names']=dct['feature_names'].tolist()
dct2['DESCR']=dct['DESCR']
print(dct2)

# Storing above data in mongodb
client = pymongo.MongoClient("mongodb+srv://gaikwadujg:rUns6cK8ABSmUpxs@cluster0.7chcxpg.mongodb.net/?retryWrites=true&w=majority")
print(client)
db = client['Project']
coll_create = db['Cancer']
coll_create.insert_one(dct2)
print('Data Successfully Imported from python to mongoDB')