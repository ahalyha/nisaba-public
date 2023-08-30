import pinecone
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

pinecone.init(
    api_key="d37388e3-0b31-4fa3-b810-599867aee2a4",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
loader = SitemapLoader(
    "https://sovren.com/sitemap.xml",
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)
docs_chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
index_name = "ind"

#create a new index
docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)