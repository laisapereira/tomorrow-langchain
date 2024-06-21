from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

load_dotenv()

def get_pdfs_content(pdf_urls):
    content = []
    for url in pdf_urls:
        loader = PyPDFLoader(url)
        pages = loader.load()
        pages_content = [page.page_content for page in pages]
        content.extend(pages_content)
    return content

pdf_urls = [
    "https://arxiv.org/pdf/2210.07342",
    "https://www.fcav.unesp.br/Home/departamentos/tecnologia/LUCIANAMARIASARAN/nocoes-de-quimica-organica.pdf",
    "https://www.seduc.ce.gov.br/wp-content/uploads/sites/37/2014/07/biotecnologia_quimica_organica.pdf",
    "https://www.ufrgs.br/colegiodeaplicacao/wp-content/uploads/2020/11/INTRODUC%CC%A7A%CC%83O-A%CC%80-QUI%CC%81MICA-ORGA%CC%82NICA.pdf",
    "https://www3.unicentro.br/petquimica/wp-content/uploads/sites/39/2018/09/Material-qu%C3%ADmica-org%C3%A2nica.pdf",
    "http://www.etelg.com.br/paginaete/downloads/ensinomedio/qu%C3%ADmica_org%C3%A2nica.pdf",
    "https://www.professores.uff.br/screspo/wp-content/uploads/sites/127/2017/09/ia_intro.pdf",
    "https://www.ufsm.br/app/uploads/sites/563/2019/09/12.4.pdf",
    "https://itsrio.org/wp-content/uploads/2019/03/Paula-Gorzoni.pdf",
    "https://repositorio.usp.br/directbitstream/71a8367e-3bce-4e42-bb52-84aba56cd43e/3041919.pdf",
    "https://educamidia.org.br/api/wp-content/uploads/2023/06/01-Palavra-Aberta-A-inteligencia-artificial-DIGITAL.pdf",
    "https://www.scielosp.org/article/ssm/content/raw/?resource_ssm_path=/media/assets/csc/v26n6/en_1413-8123-csc-26-06-2035_sxdc3lt.pdf",
    "https://repositorio.enap.gov.br/bitstream/1/7871/1/M%C3%B3dulo%201%20-%20Contextualiza%C3%A7%C3%A3o%20e%20conceitos%20sobre%20intelig%C3%AAncia%20artificial%20%28IA%29.pdf",
]

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ".",
        "!",
        "?",
        ";",
        " ",
        "",
    ],
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

all_text = get_pdfs_content(pdf_urls)
docs = text_splitter.split_text("\n".join(all_text))

for doc in docs[:5]:  # Exibe os primeiros 5 chunks como exemplo
    print("->", doc)
    print(text_splitter)
print(f"Total chunks: {len(docs)}")

documents = [{'content': doc} for doc in docs]

# Adicionar os documentos ao ChromaDB
db = Chroma.from_documents(documents,
                           OpenAIEmbeddings(),
                           persist_directory="./tomorrow")