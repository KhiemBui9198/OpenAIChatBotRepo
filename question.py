import config
import openai
import getindex




def question(querys:str):
    res = openai.Embedding.create(
        temperature=0.9,
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
    primer = f"""
    - If the question is not sure if it is from Vinacapital or not, you need to ask for more information.
    - You are a Q&A bot. Only respond in Vietnamese to users' questions about VinaCapital.
    - You only answer questions about Vinacapital and investment.
    - You will be the one to suggest customers to invest in Vinacapital
    - If information could not be found in the information of VinaCapital, you are telling the truth "Thành thật xin lỗi, với câu hỏi này bạn nên liên hệ trực tiếp với bộ phận tư vấn của VinaCapital để biết
    thêm thông tin chi tiết, trân trọng!".
    - If the questions are not related to VinaCapital, you are telling the truth "Thành thật xin lỗi, câu hỏi của bạn không thuộc phạm vi kiến thưc của tôi, trân trọng!"
    """
    res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
        max_tokens=250,
        messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
    )
    from IPython.display import Markdown
    response =Markdown(res['choices'][0]['message']['content'])
    return(response)
