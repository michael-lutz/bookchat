from keybert import KeyBERT
import re
import openai
import os
from sentence_transformers import SentenceTransformer


# All of this should be in a class

# First need a way to get keyword of input

# Then need a way to get most relevant paragraphs by keyword filtering

# Then need a way to get most relevant among those relevant paragraphs w/ embeddings

# Then feed in the most important paragraphs to the model

# Then create the prompt and get the response


openai.api_key = os.getenv('OPENAI_API_KEY')

class BookChat:
    def __init__(self, pdf_path):
        self.pdf = self._load_pdf(pdf_path)
        self.minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.maxtoken = 2400

        

    def ask(self, question, conversation_history = []):
        """Ask a question and get a response + updated conversation history"""
        self.blacklist = set() # This is kind of a sus way of approaching this...
        keywords = [x[0] for x in self._get_keywords(question)]
        relevant_paragraphs = self._get_relevant_paragraphs(keywords)
        relevant_paragraphs = self._calculate_embeddings(relevant_paragraphs, question)
        prompt = self._create_prompt(relevant_paragraphs, question, conversation_history)
        response = self._request_response(prompt)

        conversation_history.append(response)
        
        return response, conversation_history
        

    def _load_pdf(self, pdf_path):
        """Load pdf into memory"""
        with open(pdf_path, 'r') as f:
            return f.read()

    def _get_keywords(self, text):
        """Get keywords from text using keybert"""
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, top_n = 3)
        return keywords

    def _get_relevant_paragraphs(self, keywords):
        """Get relevant paragraphs from pdf"""
        paragraphs = []
        for keyword in keywords:
            paragraphs.extend(self._find_paragraph_for_term(keyword, self.pdf)) # TODO: find a way to filter out duplicates
        return paragraphs
            

    def _closest_period(self, index, direction, text):
        """Find the closest period to the index"""
        index += direction #to avoid if we are already on a period
        while index >= 0 and index < len(text) and text[index] != '.':
            self.blacklist.add(index)
            index += direction
        return index

    def _find_period(self, index, direction, text, count):
        """Find the period which is count periods away from the index"""
        j = self._closest_period(index, direction, text)
        for _ in range(count - 1):
            j = self._closest_period(j, direction, text)
        return j

    def _find_paragraph_for_term(self, term, text, period_count = 2, limit=1000, blacklist=[]):
        """Find up to <limit> paragraphs for a given term"""
        l = [m.start() for m in re.finditer(term, text.lower())]
        print(term)
        paragraphs = []
        for i in range(min(len(l) - 1, limit)):
            if l[i] in self.blacklist:
                continue

            # Find the start and end of the paragraph
            previousPeriodIndex = self._find_period(l[i], -1, text, period_count) + 1
            nextPeriodIndex = self._find_period(l[i], 1, text, period_count) + 1
            sentence = text[previousPeriodIndex:nextPeriodIndex].replace('\n', ' ')
            paragraphs.append(sentence)
            # Skip over any other instances of the term in the same paragraph
            while i + 1 < len(l) and nextPeriodIndex >= l[i + 1]:
                i += 1
        return paragraphs

    def _calculate_embeddings(self, paragraphs, question):
        """Calculate embeddings for paragraphs and return paragraphs in terms of relevance"""
        embeddings = self.minilm.encode(paragraphs)
        input_emb = self.minilm.encode(question)

        similarity = embeddings @ input_emb

        zipped = list(zip(paragraphs, similarity))
        res = [x[0] for x in sorted(zipped, key = lambda x: x[1], reverse=True)]
        return res

    def _create_prompt(self, paragraphs, question, conversation_history):
        """Create the prompt by stuffing as many paragraphs in as possible"""
        prompt = "You are a chatbot who will use the above to help answer the user's questions by paraphrasing source material. Use proper grammar.\n"
        for log in conversation_history:
            prompt += log
            prompt += "\n"
        suffix = ' Write a long and eloquent essay response.'
        prompt += "[User] " + question + suffix + "\n"
        prompt += "[You] "
        
        for paragraph in paragraphs:
            new_par = paragraph
            if len(prompt) + len(new_par) < self.maxtoken:
                prompt = new_par + "\n" + prompt

        print(prompt)
        return prompt
    
    def _request_response(self, prompt):
        """Request response from OpenAI's API"""
        response = openai.Completion.create(
          model="text-davinci-003",
          prompt=prompt,
          temperature=1,
          max_tokens=1500,
          top_p=1,
          frequency_penalty=1,
          presence_penalty=.1
        )
        response_text = response["choices"][0]["text"]
        
        return response_text

if __name__ == '__main__':
    chat = BookChat('gat_scraped.txt')

    conversation_history = []
    while True:
        txt = input("[User]: ")
        res, conversation_history = chat.ask(txt, conversation_history)
        print(res)