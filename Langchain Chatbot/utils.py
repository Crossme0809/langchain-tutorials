from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
openai.api_key = ""
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='', environment='us-east-1-aws')
index = pinecone.Index('langchain-chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"给出以下用户查询和对话记录，制定一个最相关的问题，从知识库中为用户提供一个答案.\n\n对话记录: \n{conversation}\n\n用户查询: {query}\n\n优化查询:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
