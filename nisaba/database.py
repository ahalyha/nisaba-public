import pinecone
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

pinecone.init(
    api_key="d37388e3-0b31-4fa3-b810-599867aee2a4",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
# TODO: move to the other file
# loader = SitemapLoader(
#     "https://www.textkernel.com/sitemap_index.xml"
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1200,
#     chunk_overlap  = 200,
#     length_function = len,
# )
#docs_chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
index_name = "ind"

# #create a new index
#docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)


# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)

from langchain.llms import OpenAI
llm=OpenAI()
qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)


def ask_a_question(request):
    embeddings = OpenAIEmbeddings()
    index_name = "ind"
   
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    llm=OpenAI()
    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    result = qa_with_sources({"query": "Give please very polite ansver to a question as you are a helpfull assistant"+request})
    return format_slack_message(result)

def format_slack_message(response):
    result = response['result']
    source_documents = response['source_documents']

    message = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*What I see:*\n{result}"
                }
            }
        ]
    }

    link_count = 0

    for document in source_documents:
        if link_count >= 2:  # Break loop if 2 links are added
            break
        
        metadata = document.metadata

        if 'loc' in metadata:
            link = metadata['loc']
            message['blocks'].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<{link}|{link}>"
                }
            })

            link_count += 1  # Increment the link count
    return message


print(ask_a_question("Tell me about Textkernel products"))
