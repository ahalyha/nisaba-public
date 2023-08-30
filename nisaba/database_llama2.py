import pinecone
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import cuda
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.llms import HuggingFacePipeline

from dotenv import load_dotenv, find_dotenv
from torch import cuda, bfloat16
import transformers



_ = load_dotenv(find_dotenv())

# pinecone.init(
#     api_key="d37388e3-0b31-4fa3-b810-599867aee2a4",  # find at app.pinecone.io
#     environment="gcp-starter"  # next to api key in console
# )

pinecone.init(
    api_key='a324de21-f3bd-4f11-b258-bf6b889877c2',
    environment='us-west1-gcp-free'
)

# TODO: move to the other file
# loader = SitemapLoader(
#     "https://www.textkernel.com/sitemap_index.xml"
# )
# docs = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1200,
#     chunk_overlap  = 200,
#     length_function = len,
# )
# docs_chunks = text_splitter.split_documents(docs)

# create embedding model
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
embeddings = embed_model

# create hugging face pipeline

model_id = 'meta-llama/Llama-2-13b-chat-hf'
#model_id = 'Tap-M/Luna-AI-Llama2-Uncensored'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)

index_name = 'llama-2-rag'

# #create a new index

# docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)


# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                              return_source_documents=True)

def ask_a_question(request):

    result = qa_with_sources({"query": request})
    return format_slack_message(result)


def format_slack_message(response):
    query = response['query']
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

    for document in source_documents:
        page_content = document.page_content
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

    return message


print(ask_a_question("Tell me about Textkernel products"))
