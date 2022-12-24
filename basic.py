import os
import openai

openai.api_key = os.getenv("OPENAI_KEY")

# response = openai.Completion.create(
#   model="text-davinci-003",
#   prompt="Brainstorm some ideas combining VR and fitness:",
#   temperature=0.6,
#   max_tokens=150,
#   top_p=1,
#   frequency_penalty=1,
#   presence_penalty=1
# )

# 1536 long array

response = openai.Embedding.create(
    input="Your text string goes here",
    model="text-embedding-ada-002"
)
embeddings = response['data'][0]['embedding']
print(response)
import pdb; pdb.set_trace()