"""
================================================================================
MD文档读取器模块 (md_document_reader.py)

【这个文件的作用】
专门处理MD文件的读取和解析

【核心功能】
1. 读取MD文件内容
2. 根据Markdown标题结构进行智能分块
3. 支持三级标题分块，内容过多时按字符数分块
4. 返回适合检索的文本片段

【依赖库】
    pip install langchain-text-splitters
    pip install langchain-community
================================================================================
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割工具
from langchain_community.document_loaders import TextLoader  # 文本加载器

from document_reader_base import DocumentReaderBase
from logger import logger


class MDDocumentReader(DocumentReaderBase):
    """
    【类名】MDDocumentReader
    
    【作用】专门处理MD和TXT文件的读取和解析
    
    【支持的格式】
        1. .md (Markdown格式)
        2. .txt (纯文本格式)
    
    【设计理念】
        1. 专注于文本文件解析逻辑
        2. 继承基类的统一接口和工具方法
        3. 提供详细的日志和错误处理
        4. 根据Markdown标题结构进行智能分块
        5. 对于纯文本文件按字符数分块
    """
    
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表
        """
        return ['.md', '.txt']
    
    def read_document(self, filepath):
        """
        【功能】从MD或TXT文件中提取内容
        
        【参数】
            filepath: MD或TXT文件的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 验证文件是否存在且可读
            2. 检查文件扩展名是否支持
            3. 读取MD或TXT文件内容   
            4. 根据Markdown标题结构进行智能分块
            5. 对过长的内容按字符数分块
            
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
            # 读取MD文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据Markdown标题结构进行智能分块
            chunks = self._split_md_content(content)
            
            # 计算总字符数
            total_chars = sum(len(chunk) for chunk in chunks)
            self._log(f'MD文件共 {total_chars} 个字符')
            self._log(f'切分成 {len(chunks)} 个片段')
            
            return chunks
            
        except Exception as e:
            self._log_error(f'读取文件 {filepath} 时出错: {e}')
            return []
    
    def _split_md_content(self, content, chunk_size=500, overlap=100):
        """
        【功能】根据Markdown标题结构进行智能分块
        
        【参数】
            content: MD文件内容
            chunk_size: 最大块大小（默认500字符）
            overlap: 块重叠大小（默认100字符）
            
        【返回值】
            分割后的文本片段列表
            
        【处理逻辑】
            1. 识别一级标题和二级标题
            2. 按三级标题（###）分割内容
            3. 对每个三级标题下的内容，添加上级标题信息
            4. 对每个标题下的内容，如果超过chunk_size则进一步分块
            5. 保持标题和内容的完整性
        """
        chunks = []
        
        # 识别一级标题和二级标题
        current_h1 = ""
        current_h2 = ""
        sections = []
        
        # 按行处理内容，记录当前的标题层级
        lines = content.split('\n')
        current_section = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 检测一级标题
            if line_stripped.startswith('# '):
                if current_section:
                    sections.append({"h1": current_h1, "h2": current_h2, "content": '\n'.join(current_section)})
                    current_section = []
                current_h1 = line_stripped[2:].strip()
                current_h2 = ""
            # 检测二级标题
            elif line_stripped.startswith('## '):
                if current_section:
                    sections.append({"h1": current_h1, "h2": current_h2, "content": '\n'.join(current_section)})
                    current_section = []
                current_h2 = line_stripped[3:].strip()
            # 检测三级标题
            elif line_stripped.startswith('### '):
                if current_section:
                    sections.append({"h1": current_h1, "h2": current_h2, "content": '\n'.join(current_section)})
                    current_section = []
                current_section.append(line)
            else:
                current_section.append(line)
        
        # 添加最后一个部分
        if current_section:
            sections.append({"h1": current_h1, "h2": current_h2, "content": '\n'.join(current_section)})
        
        # 处理每个部分
        for section in sections:
            h1 = section["h1"]
            h2 = section["h2"]
            section_content = section["content"]
            
            if not section_content.strip():
                continue
            
            # 构建完整的内容，包含上级标题
            full_content = []
            if h1:
                full_content.append(f"# {h1}")
            if h2:
                full_content.append(f"## {h2}")
            full_content.append(section_content)
            
            full_content_str = '\n'.join(full_content)
            
            # 检查内容长度
            if len(full_content_str) <= chunk_size:
                chunks.append(full_content_str)
            else:
                # 使用RecursiveCharacterTextSplitter进一步分块
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                sub_chunks = splitter.split_text(full_content_str)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def extract_metadata(self, filepath):
        """
        【功能】提取MD文件的元数据（扩展功能）
        
        【参数】
            filepath: MD文件的文件路径
            
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
        python md_document_reader.py
    """
    logger.info("=" * 100)
    logger.info("MDDocumentReader 模块测试")
    logger.info("=" * 100)
    
    # 创建MD文档读取器实例
    reader = MDDocumentReader()
    
    # 测试支持的扩展名
    logger.info(f"支持的扩展名: {reader.get_supported_extensions()}")
    
    # 测试文件类型检测
    test_files = [
        "test.md",       # 支持的文件
        "test_data.md",  # 支持的文件
        "test.docx",     # 不支持的文件
        "test.pdf",      # 不支持的文件
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
    test_md_file = "test_data.md"
    
    if os.path.exists(test_md_file):
        documents = reader.read_document(test_md_file)
        logger.info(f"\n读取文件: {test_md_file}")
        logger.info(f"  读取结果: {len(documents)} 个片段")
        
        # 打印所有分块内容
        if documents:
            logger.info("\n分块内容:")
            for i, chunk in enumerate(documents, 1):
                logger.info(f"\n--- 片段 {i} ---")
                logger.info(chunk)
    else:
        logger.info(f"\n测试文件 {test_md_file} 不存在")
    
    logger.info("\n" + "=" * 100)
    logger.info("测试完成")
    logger.info("=" * 100)
