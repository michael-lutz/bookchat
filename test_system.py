
from pymongo import MongoClient
import openai
import numpy as np
import os

"""
Limits on token length

The number of tokens processed in a given API request depends on the length of both your inputs and outputs. 
As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text. 
One limitation to keep in mind is that your text prompt and generated completion combined must be no more than 
the model's maximum context length (for most models this is 2048 tokens, or about 1500 words)
"""

openai.api_key = os.getenv("OPENAI_KEY")

def sort_documents(db, text):
    data = []
    X = []
    for post in db.Article.find():
        data.append(post)
        X.append(np.array(post["embedding"]))
    
    # X = rows of data X 1536
    X = np.array(X)
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = np.array(response['data'][0]['embedding'])

    # num rows of data X 1
    o = X @ embedding   
    l  = [(v, i) for i, v in enumerate(o)]
    l.sort(key=lambda x: x[0])

    sorted = []
    for i in range(len(data)):
        sorted.append(data[l[-(i+1)][1]])
    return sorted


class ChatAssist:
    def __init__(self) -> None:
        self.db = self._get_database()
        self.conversation_buffer = []
        self.MAX_CHARS = 7500 # See limits on token length

    def converse(self, text):
        response = None
        docs = sort_documents(self.db, text)
        
        # Construct prompt
        self.conversation_buffer.append(f"[User]: {text}")
        convo = self._compile_convo()

        prompt = ""
        prompt_end = "You are a chatbot customer support agent for a company and should continue the conversation in a cordial and professional manner using the information provided above to guide your responses.\n"
        for i in range(len(docs)):
            new_content = docs[i]["title"] + ":\n" + docs[i]["summarizedContent"] + "\n"
            if len(new_content) + len(prompt) + len(prompt_end) + len(convo) > self.MAX_CHARS:
                 break
            prompt += new_content
        final_prompt = prompt  + prompt_end + convo + "[agent]:"

        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=final_prompt,
          temperature=0.6,
          max_tokens=2048,
          top_p=1,
          frequency_penalty=1,
          presence_penalty=1
        )
        response_text = response["choices"][0]["text"]
        self.conversation_buffer.append("[agent]:" + response_text)

        return response_text

    def _compile_convo(self):
        s = ""
        for c in self.conversation_buffer:
            s += c + "\n"
        return s

    def _get_database(self):
        CONNECTION_STRING = os.getenv("CONNECTION_STRING")
        client = MongoClient(CONNECTION_STRING)
        return client['chatassist-development']

if __name__ == "__main__":
    bot =  ChatAssist()
    # bot.converse("Hello I am having issues logging into the website. What should I do?")
    while True:
        txt = input("[User]: ")
        print(bot.converse(txt))
