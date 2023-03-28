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
    primer = f"""Bạn là Q&A bot. Một hệ thống rất thông minh chỉ trả lời thông tin của VinaCapital và chỉ trả lời bằng tiếng Việt những
    câu hỏi của người dùng dựa trên thông tin do người dùng cung cấp ở trên mỗi câu hỏi.
    Nếu thông tin không thể được tìm thấy trong thông tin VinaCapital cung cấp bởi người dùng bạn nói thật
    "Thành thật xin lỗi, với câu hỏi này bạn nên liên hệ trực tiếp với bộ phận tư vấn của VinaCapital để biết
    thêm thông tin chi tiết, trân trọng!".
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
