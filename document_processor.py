#!/usr/bin/env python3
"""
================================================================================
文档处理类模块 (document_processor.py)

【功能】
- 统一处理不同类型文档的加载、分割、摘要生成和索引
- 支持PDF和Word文档
- 使用摘要索引和多向量检索器提升检索质量
================================================================================
"""

import os
import hashlib
import uuid
import chromadb
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiVectorRetriever

from logger import logger
from document_reader_pdf import PDFDocumentReader
from document_reader_word import WordDocumentReader
from document_reader_csv import CSVDocumentReader
from document_reader_md import MDDocumentReader
from document_reader_xlsx import XLSXDocumentReader
from models import get_ali_embeddings, get_ds_model_client


#!/usr/bin/env python3
"""
================================================================================
文档处理类模块 (document_processor.py)

【功能】
- 统一处理不同类型文档的加载、分割、摘要生成和索引
- 支持PDF和Word文档
- 使用摘要索引和多向量检索器提升检索质量
================================================================================
"""

def clean_collection_name(filename: str) -> str:
    """
    【功能】清理文件名或集合名，使其符合ChromaDB集合名称规范
    """
    import re

    if '.' not in filename:
        if 3 <= len(filename) <= 63:
            if re.match(r'^[a-zA-Z0-9_-]+$', filename):
                if filename[0].isalnum() and filename[-1].isalnum():
                    if '..' not in filename:
                        if not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', filename):
                            return filename

    name_without_ext = os.path.splitext(filename)[0]
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', name_without_ext)

    if not cleaned:
        cleaned = 'doc'

    cleaned = cleaned.strip('-_')

    if not cleaned:
        cleaned = 'doc'

    max_base_length = 57
    if len(cleaned) > max_base_length:
        cleaned = cleaned[:max_base_length]

    file_hash = hashlib.md5(filename.encode()).hexdigest()[:6]
    result = f"{cleaned}_{file_hash}"

    if len(result) > 63:
        max_allowed = 63 - len(file_hash) - 1
        cleaned = cleaned[:max_allowed]
        result = f"{cleaned}_{file_hash}"

    if not result[0].isalnum():
        result = 'doc_' + result[1:] if len(result) > 1 else 'doc'
    if not result[-1].isalnum():
        result = result[:-1] + 'x' if len(result) > 1 else 'doc'

    if len(result) < 3:
        result = 'doc_' + file_hash

    return result


class DocumentProcessor:
    """
    文档处理类
    使用摘要索引和多向量检索器提升检索质量
    """

    def __init__(self):
        self.client = get_ds_model_client()
        self.embeddings_model = get_ali_embeddings()
        self.pdf_reader = PDFDocumentReader()
        self.word_reader = WordDocumentReader()
        self.csv_reader = CSVDocumentReader()
        self.md_reader = MDDocumentReader()
        self.xlsx_reader = XLSXDocumentReader()
        self.processed_docs = {}
        self.chroma_client = chromadb.PersistentClient(path="./chroma")

    def get_supported_extensions(self):
        pdf_extensions = self.pdf_reader.get_supported_extensions()
        word_extensions = self.word_reader.get_supported_extensions()
        csv_extensions = self.csv_reader.get_supported_extensions()
        md_extensions = self.md_reader.get_supported_extensions()
        xlsx_extensions = self.xlsx_reader.get_supported_extensions()
        return list(set(pdf_extensions + word_extensions + csv_extensions + md_extensions + xlsx_extensions))

    def supports_extension(self, extension):
        return extension.lower() in self.get_supported_extensions()

    def _generate_summaries(self, split_docs):
        logger.info("准备生成文档摘要...")
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("""请仔细总结下面的文档内容，确保保留所有关键信息。

            【重要】
            1. 如果是表格，要保留完整的表格结构和所有行列数据，不要遗漏任何单元格！
            2. 表格中的每一行每一列都必须完整保留！
            3. 不要总结或合并数据，要原样保留！

            文档内容:
            {doc}""")
            | self.client
            | StrOutputParser()
        )
        summaries = chain.batch(split_docs, {"max_concurrency": 5})
        logger.info(f"摘要生成完成，获得 {len(summaries)} 个摘要")
        return summaries

    def _build_multi_vector_index(self, split_docs, summaries, source_file, collection_name):
        logger.info(f"准备清空集合 '{collection_name}' 中的旧数据...")
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"集合 '{collection_name}' 已清空")
        except Exception as e:
            logger.warning(f"清空集合时出错（可能不存在）: {e}")

        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embeddings_model
        )

        store = InMemoryByteStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )

        doc_ids = [str(uuid.uuid4()) for _ in split_docs]

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i], "source_file": source_file, "type": "summary"})
            for i, s in enumerate(summaries)
        ]

        logger.info("准备将摘要存入向量数据库...")
        retriever.vectorstore.add_documents(summary_docs)

        logger.info("准备将原始文档存储到字节存储...")
        retriever.docstore.mset(list(zip(doc_ids, split_docs)))

        logger.info("准备将原始文档也存入向量数据库...")
        retriever.vectorstore.add_documents(split_docs)

        return retriever

    def _process_document(self, filepath, reader, doc_type, collection_name=None):
        logger.info(f"处理{doc_type}文档: {filepath}")

        raw_docs = reader.read_document(filepath)
        if not raw_docs:
            logger.error(f"{doc_type}文档处理失败")
            return None

        logger.info(f"原始文档数量: 1")
        logger.info(f"分割后文档片段数量: {len(raw_docs)}")

        source_file = os.path.basename(filepath)
        split_docs = [Document(page_content=doc, metadata={"source_file": source_file}) for doc in raw_docs]

        summaries = self._generate_summaries(split_docs)

        if collection_name is None:
            collection_name = clean_collection_name(os.path.basename(filepath))

        retriever = self._build_multi_vector_index(split_docs, summaries, source_file, collection_name)

        result = {
            'filepath': filepath,
            'type': doc_type.lower(),
            'split_docs': split_docs,
            'summaries': summaries,
            'retriever': retriever,
            'collection_name': collection_name
        }
        self.processed_docs[filepath] = result
        return result

    def process_document(self, filepath, collection_name=None):
        extension = os.path.splitext(filepath)[1].lower()

        if not self.supports_extension(extension):
            logger.error(f"不支持的文件类型: {extension}")
            return None

        reader_map = {
            '.pdf': (self.pdf_reader, 'PDF'),
            '.docx': (self.word_reader, 'Word'),
            '.doc': (self.word_reader, 'Word'),
            '.csv': (self.csv_reader, 'CSV'),
            '.md': (self.md_reader, 'MD'),
            '.txt': (self.md_reader, '文本'),
            '.xlsx': (self.xlsx_reader, 'Excel'),
        }

        reader, doc_type = reader_map.get(extension, (None, None))
        if reader is None:
            logger.error(f"未知文件类型: {extension}")
            return None

        return self._process_document(filepath, reader, doc_type, collection_name)

    def _process_pdf(self, filepath, collection_name=None):
        return self._process_document(filepath, self.pdf_reader, 'PDF', collection_name)

    def _process_word(self, filepath, collection_name=None):
        return self._process_document(filepath, self.word_reader, 'Word', collection_name)

    def _process_csv(self, filepath, collection_name=None):
        return self._process_document(filepath, self.csv_reader, 'CSV', collection_name)

    def _process_md(self, filepath, collection_name=None):
        return self._process_document(filepath, self.md_reader, 'MD', collection_name)

    def _process_xlsx(self, filepath, collection_name=None):
        return self._process_document(filepath, self.xlsx_reader, 'Excel', collection_name)

    def query_document(self, filepath, query):
        """
        使用多向量检索器查询文档
        """
        from function_tools import get_completion

        if filepath not in self.processed_docs:
            logger.error(f"文档未处理: {filepath}")
            return None

        result = self.processed_docs[filepath]
        retriever = result['retriever']

        retrieved_docs = retriever.invoke(query)
        info = '\n'.join([doc.page_content for doc in retrieved_docs])

        # 构建提示词的逻辑已移至 get_completion 函数内部
        response = get_completion(info, query)

        return {
            'answer': response,
            'retrieved_docs': retrieved_docs
        }


if __name__ == '__main__':
    logger.info("=" * 100)
    logger.info("DocumentProcessor 模块测试")
    logger.info("=" * 100)

    processor = DocumentProcessor()
    logger.info(f"支持的扩展名: {processor.get_supported_extensions()}")

