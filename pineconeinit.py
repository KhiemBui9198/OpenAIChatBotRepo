import pinecone

def pineconeinit(res:str):
    index_name = 'gpt-4-langchain-docs'

# initialize connection to pinecone
    pinecone.init(
        api_key = "9abf60de-a00a-4302-abfa-8d5279584f38",
        environment = "us-central1-gcp"
    )
# check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
    # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=len(res['data'][0]['embedding']),
            metric='dotproduct'
        )
# connect to index
    index = pinecone.GRPCIndex(index_name)
# view index stats
    return index