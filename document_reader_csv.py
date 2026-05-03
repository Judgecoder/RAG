"""
================================================================================
CSV文档读取器模块 (csv_document_reader.py)

【这个文件的作用】
专门处理CSV文件的读取和解析

【核心功能】
1. 使用langchain的CSVLoader读取CSV文件
2. 提取所有行数据并转换为文本格式
3. 使用智能文本分割器将长文档切分
4. 返回适合检索的文本片段

【依赖库】
    pip install langchain-text-splitters
    pip install langchain-community
================================================================================
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割工具
from langchain_community.document_loaders import CSVLoader  # CSV加载器

from document_reader_base import DocumentReaderBase
from logger import logger


class CSVDocumentReader(DocumentReaderBase):
    """
    【类名】CSVDocumentReader
    
    【作用】专门处理CSV文件的读取和解析
    
    【支持的格式】
        1. .csv (逗号分隔值格式)
    
    【设计理念】
        1. 专注于CSV文档解析逻辑
        2. 继承基类的统一接口和工具方法
        3. 提供详细的日志和错误处理
    """
    
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表
        """
        return ['.csv']
    
    def read_document(self, filepath):
        """
        【功能】从CSV文件中提取内容
        
        【参数】
            filepath: CSV文件的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 验证文件是否存在且可读
            2. 检查文件扩展名是否支持
            3. 使用CSVLoader加载CSV文件
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
            # 使用CSVLoader加载文档
            loader = CSVLoader(filepath)
            docs = loader.load()
            
            # 使用RecursiveCharacterTextSplitter切分文本
            # 与其他读取器保持一致的参数
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            # split_documents() 执行切分
            split_docs = text_splitter.split_documents(docs)
            
            # 提取切分后的文本内容
            documents = [doc.page_content for doc in split_docs]
            
            # 计算总字符数
            total_chars = sum(len(doc) for doc in documents)
            self._log(f'CSV文件共 {total_chars} 个字符')
            self._log(f'切分成 {len(documents)} 个片段')
            
            return documents
            
        except Exception as e:
            self._log_error(f'读取文件 {filepath} 时出错: {e}')
            return []
    
    def extract_metadata(self, filepath):
        """
        【功能】提取CSV文件的元数据（扩展功能）
        
        【参数】
            filepath: CSV文件的文件路径
            
        【返回值】
            文档元数据字典，包含文件信息等
            
        【说明】
            这是可选功能，展示了如何扩展特定文档类型的解析能力
        """
        try:
            # 提取基本文件信息作为元数据
            file_stat = os.stat(filepath)
            
            metadata = {
                'filename': os.path.basename(filepath),
                'size': file_stat.st_size,
                'created': file_stat.st_ctime,
                'modified': file_stat.st_mtime,
                'path': filepath
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
        python csv_document_reader.py
    """
    logger.info("=" * 100)
    logger.info("CSVDocumentReader 模块测试")
    logger.info("=" * 100)
    
    # 创建CSV文档读取器实例
    reader = CSVDocumentReader()
    
    # 测试支持的扩展名
    logger.info(f"支持的扩展名: {reader.get_supported_extensions()}")
    
    # 测试文件类型检测
    test_files = [
        "test.csv",       # 支持的文件
        "test_data.csv",  # 支持的文件
        "test.docx",      # 不支持的文件
        "test.pdf",       # 不支持的文件
    ]
    
    for test_file in test_files:
        logger.info(f"\n测试文件: {test_file}")
        extension = reader._get_file_extension(test_file)
        supported = reader.supports_extension(extension)
        logger.info(f"  扩展名: {extension}, 支持: {supported}")
    
    # 测试读取功能
    logger.info("\n" + "=" * 100)
    logger.info("测试读取功能")
    logger.info("=" * 100)
    
    test_csv_file = "test_data.csv"
    if os.path.exists(test_csv_file):
        documents = reader.read_document(test_csv_file)
        logger.info(f"\n读取文件: {test_csv_file}")
        logger.info(f"  读取结果: {len(documents)} 个片段")
        
        # 打印所有分块内容
        if documents:
            logger.info("\n分块内容:")
            for i, chunk in enumerate(documents, 1):
                logger.info(f"\n--- 片段 {i} ---")
                logger.info(chunk)
    else:
        logger.info(f"\n测试文件 {test_csv_file} 不存在")
    
    logger.info("\n" + "=" * 100)
    logger.info("测试完成")
    logger.info("=" * 100)
