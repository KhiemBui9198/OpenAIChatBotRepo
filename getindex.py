
# initialize connection to pinecone
import pinecone

def get_index():
    index_name = 'gpt-4-langchain-docs'
    pinecone.init(
    api_key = "9abf60de-a00a-4302-abfa-8d5279584f38",
    environment = "us-central1-gcp"
    )

    index = pinecone.GRPCIndex(index_name)
# view index stats
    return index