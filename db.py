from pymongo import MongoClient
import pprint
import os

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
    dbname.Article.find_one()
    i = 0
    for post in dbname.Article.find():
        i+=1
        print(post)
    print(f"Retrieved a total of {i} posts")
    import pdb; pdb.set_trace()
    # res.acknowledged 