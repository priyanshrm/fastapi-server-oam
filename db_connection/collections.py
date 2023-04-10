from pymongo import MongoClient
import certifi

CONNECTION_STR =  "mongodb+srv://priyanshmahendra:8OUaa5MDTILL8KXc@cluster1.0aj9lo3.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STR, tlsCAFile=certifi.where())
# client = MongoClient("mongodb://localhost:27017")
db = client["social_media"]
images_db = client["test_db"]
artistsCollection = db["artists"]
collectorCollection = db["collectors"]
encodingCollection = db["encodings"]