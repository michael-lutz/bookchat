from pymongo import MongoClient
import openai
from tqdm import tqdm
import os

openai.api_key = os.getenv("OPENAI_KEY")

def get_database():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = os.getenv("CONNECTION_STRING")
   
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)
 
   # Create the database for our example (we will use the same database throughout the tutorial
   return client['chatassist-development']
  
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":   
    # Get the database
    dbname = get_database()
    for post in tqdm(dbname.Article.find()):
        if "embedding" not in post.keys():
            text = post["summarizedContent"]
            id = post["_id"]
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            embedding = response['data'][0]['embedding']

            awk = False
            while not awk:
                res = dbname.Article.update_one({"_id": id}, {"$set": {"embedding": embedding}})
                awk = res.acknowledged