import config
import openai
import getindex




def question(querys:str):
    res = openai.Embedding.create(
        input=[querys],
        engine=config.embed_model
    )
# retrieve from Pinecone
    xq = res['data'][0]['embedding']
    index = getindex.get_index()
# get relevant contexts (including the questions)
    res = index.query(xq, top_k=10, include_metadata=True)
    contexts = [item['metadata']['text'] for item in res['matches']]

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+querys
# system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    #provided by the user you truthfully say "I don't know and Just answer by vietnamese".
    """
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
    )
    from IPython.display import Markdown
    response =Markdown(res['choices'][0]['message']['content'])
    return(response)