"""
================================================================================
XLSX文档读取器模块 (xlsx_document_reader.py)

【这个文件的作用】
专门处理XLSX文件的读取和解析

【核心功能】
1. 使用langchain的ExcelLoader读取XLSX文件
2. 提取所有工作表数据并转换为文本格式
3. 使用智能文本分割器将长文档切分
4. 返回适合检索的文本片段

【依赖库】
    pip install langchain-text-splitters
    pip install langchain-community
    pip install openpyxl  # 用于Excel文件解析
================================================================================
"""

import os
import pandas as pd

from document_reader_base import DocumentReaderBase
from logger import logger


class XLSXDocumentReader(DocumentReaderBase):
    """
    【类名】XLSXDocumentReader
    
    【作用】专门处理XLSX文件的读取和解析
    
    【支持的格式】
        1. .xlsx (Excel文件格式)
    
    【设计理念】
        1. 专注于XLSX文档解析逻辑
        2. 继承基类的统一接口和工具方法
        3. 提供详细的日志和错误处理
    """
    
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表
        """
        return ['.xlsx']
    
    def read_document(self, filepath):
        """
        【功能】从XLSX文件中提取内容
        
        【参数】
            filepath: XLSX文件的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 验证文件是否存在且可读
            2. 检查文件扩展名是否支持
            3. 使用pandas读取XLSX文件
            4. 按行进行分块，每行都附带上表头
            
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
            excel_file = pd.ExcelFile(filepath)
            sheet_names = excel_file.sheet_names
            self._log(f'检测到 {len(sheet_names)} 个工作表: {sheet_names}')

            documents = []
            chunk_size = 1000

            for sheet_name in sheet_names:
                df = excel_file.parse(sheet_name=sheet_name)
                headers = list(df.columns)

                for index, row in df.iterrows():
                    row_with_header = []
                    for col in headers:
                        value = row[col] if col in df.columns else ''
                        row_with_header.append(f"{col}: {value}")
                    row_with_header = '\n'.join(row_with_header)

                    sheet_marker = f"[来源工作表: {sheet_name}] "
                    row_with_marker = sheet_marker + row_with_header

                    row_size = len(row_with_marker)

                    if row_size > chunk_size:
                        self._log_warning(f'工作表 {sheet_name} 的行 {index+1} 超过chunk_size，可能需要进一步处理')
                        documents.append(row_with_marker[:chunk_size])
                    else:
                        documents.append(row_with_marker)

            total_chars = sum(len(doc) for doc in documents)
            self._log(f'XLSX文件共 {total_chars} 个字符')
            self._log(f'切分成 {len(documents)} 个片段')

            return documents

        except Exception as e:
            self._log_error(f'读取文件 {filepath} 时出错: {e}')
            return []
    
    def extract_metadata(self, filepath):
        """
        【功能】提取XLSX文件的元数据（扩展功能）
        
        【参数】
            filepath: XLSX文件的文件路径
            
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
        python xlsx_document_reader.py
    """
    logger.info("=" * 100)
    logger.info("XLSXDocumentReader 模块测试")
    logger.info("=" * 100)
    
    # 创建XLSX文档读取器实例
    reader = XLSXDocumentReader()
    
    # 测试支持的扩展名
    logger.info(f"支持的扩展名: {reader.get_supported_extensions()}")
    
    # 测试文件类型检测
    test_files = [
        "test.xlsx",       # 支持的文件
        "test_data.xlsx",  # 支持的文件
        "test.docx",       # 不支持的文件
        "test.pdf",        # 不支持的文件
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
    
    # 创建测试文件
    import pandas as pd
    import tempfile
    
    # 创建一个简单的Excel文件
    test_xlsx_file = "test_data.xlsx"
    
    if os.path.exists(test_xlsx_file):
        documents = reader.read_document(test_xlsx_file)
        logger.info(f"\n读取文件: {test_xlsx_file}")
        logger.info(f"  读取结果: {len(documents)} 个片段")
        
        # 打印所有分块内容
        if documents:
            logger.info("\n分块内容:")
            for i, chunk in enumerate(documents, 1):
                logger.info(f"\n--- 片段 {i} ---")
                logger.info(chunk)
    else:
        logger.info(f"\n测试文件 {test_xlsx_file} 不存在")
    
    logger.info("\n" + "=" * 100)
    logger.info("测试完成")
    logger.info("=" * 100)
