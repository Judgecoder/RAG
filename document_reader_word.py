"""
================================================================================
Word文档读取器模块 (word_document_reader.py)

【这个文件的作用】
专门处理Word文档（.docx, .doc）的读取和解析

【核心功能】
1. 使用python-docx库读取Word文档
2. 提取所有段落文本
3. 使用智能文本分割器将长文档切分
4. 返回适合检索的文本片段

【依赖库】
    pip install python-docx
    pip install langchain-text-splitters
================================================================================
"""

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

from document_reader_base import DocumentReaderBase
from logger import logger


class WordDocumentReader(DocumentReaderBase):
    """
    【类名】WordDocumentReader
    
    【作用】专门处理Word文档的读取和解析
    
    【支持的格式】
        1. .docx (Office Open XML格式)
        2. .doc (较旧的二进制格式，部分支持)
    
    【设计理念】
        1. 专注于Word文档解析逻辑
        2. 继承基类的统一接口和工具方法
        3. 提供详细的日志和错误处理
    """
    
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表
        """
        return ['.docx', '.doc']
    
    def read_document(self, filepath):
        """
        【功能】从Word文档中提取文字内容
        
        【参数】
            filepath: Word文档的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 验证文件是否存在且可读
            2. 检查文件扩展名是否支持
            3. 使用Docx2txtLoader加载Word文档
            4. 使用RecursiveCharacterTextSplitter切分文档
            
        【异常处理】
            - 文件不存在或不可读：返回空列表
            - 文件类型不支持：返回空列表
            - 解析过程中出错：返回空列表并记录错误
        """
        # 验证文件
        if not self._validate_file(filepath):
            return []
        
        # 检查扩展名
        extension = self._get_file_extension(filepath)
        if not self.supports_extension(extension):
            self._log_error(f'不支持的文件类型: {extension}')
            return []
        
        self._log(f'正在读取: {filepath}')
        
        try:
            # 使用Docx2txtLoader加载文档
            loader = Docx2txtLoader(filepath)
            docs = loader.load()
            
            # 使用RecursiveCharacterTextSplitter切分文本
            # 与test.py保持一致的参数
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )
            
            # split_documents() 执行切分
            split_docs = text_splitter.split_documents(docs)
            
            # 提取切分后的文本内容
            documents = [doc.page_content for doc in split_docs]
            
            self._log(f'原文共 {len(docs[0].page_content)} 个字符')
            self._log(f'切分成 {len(documents)} 个片段')
            
            return documents
            
        except Exception as e:
            self._log_error(f'读取文件 {filepath} 时出错: {e}')
            return []
    
    def extract_metadata(self, filepath):
        """
        【功能】提取Word文档的元数据（扩展功能）
        
        【参数】
            filepath: Word文档的文件路径
            
        【返回值】
            文档元数据字典，包含作者、标题、创建时间等信息
            
        【说明】
            这是可选功能，展示了如何扩展特定文档类型的解析能力
        """
        try:
            doc = Document(filepath)
            core_props = doc.core_properties
            
            metadata = {
                'author': core_props.author,
                'title': core_props.title,
                'subject': core_props.subject,
                'created': core_props.created,
                'modified': core_props.modified,
                'last_modified_by': core_props.last_modified_by,
                'revision': core_props.revision,
                'category': core_props.category,
                'comments': core_props.comments,
                'keywords': core_props.keywords,
            }
            
            self._log(f'提取元数据: {len([v for v in metadata.values() if v])} 个字段')
            return metadata
            
        except Exception as e:
            self._log_error(f'提取元数据时出错: {e}')
            return {}


# =============================================================================
# 模块测试代码
# =============================================================================
if __name__ == '__main__':
    """
    【功能】模块独立测试
    
    【使用方式】
        python word_document_reader.py
    """
    logger.info("=" * 100)
    logger.info("WordDocumentReader 模块测试")
    logger.info("=" * 100)
    
    # 创建Word文档读取器实例
    reader = WordDocumentReader()
    
    # 测试支持的扩展名
    logger.info(f"支持的扩展名: {reader.get_supported_extensions()}")
    
    # 测试文件类型检测
    test_files = [
        "test.docx",      # 支持的文件
        "test.doc",       # 支持的文件  
        "test.pdf",       # 不支持的文件
        "test.txt",       # 不支持的文件
    ]
    
    for test_file in test_files:
        logger.info(f"\n测试文件: {test_file}")
        extension = reader._get_file_extension(test_file)
        supported = reader.supports_extension(extension)
        logger.info(f"  扩展名: {extension}, 支持: {supported}")
    
    logger.info("\n" + "=" * 100)
    logger.info("测试完成")
    logger.info("=" * 100)
