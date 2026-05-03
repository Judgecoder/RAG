"""
================================================================================
PDF文档读取器模块 (pdf_document_reader.py)

【这个文件的作用】
专门处理PDF文档的读取和解析，支持两种解析模式：
1. MinerU智能解析：提供结构化内容提取（文本、表格、标题级别）
2. PyPDF2基础解析：回退方案，提取基础文本内容

【核心功能】
1. 优先尝试MinerU智能解析（如果可用）
2. 自动回退到PyPDF2基础解析
3. 智能内容分割（保持表格结构、标题级别）
4. 返回适合检索的文本片段

【依赖库】
    pip install PyPDF2
    pip install langchain-text-splitters
    pip install mineru (可选，用于智能解析)
================================================================================
"""

import os
import tempfile
import json
import shutil
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割工具

from document_reader_base import DocumentReaderBase
from logger import logger

# 尝试导入MinerU工具（用于智能PDF解析）
try:
    from mineru.cli.common import do_parse, read_fn
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

# 尝试导入PyPDF2（用于基础PDF解析）
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


class PDFDocumentReader(DocumentReaderBase):
    """
    【类名】PDFDocumentReader
    
    【作用】专门处理PDF文档的读取和解析
    
    【支持的格式】
        1. .pdf (Portable Document Format)
    
    【解析模式】
        1. 智能模式：使用MinerU工具进行结构化解析（如果可用）
        2. 基础模式：使用PyPDF2提取文本内容（回退方案）
    
    【设计理念】
        1. 智能优先：优先使用MinerU获取结构化内容
        2. 优雅降级：MinerU不可用时自动回退到PyPDF2
        3. 内容感知：保持表格结构、标题级别等语义信息
    """
    
    def __init__(self, prefer_mineru=True, save_intermediate_files=False, intermediate_output_dir=None):
        """
        【功能】初始化PDF文档读取器
        
        【参数】
            prefer_mineru: 是否优先使用MinerU智能解析（默认True）
            save_intermediate_files: 是否保存MinerU生成的中间文件（默认False）
            intermediate_output_dir: 中间文件输出目录（默认None，使用./mineru_output/）
            
        【说明】
            如果prefer_mineru为True且MinerU可用，将使用智能解析
            否则使用PyPDF2基础解析
            如果save_intermediate_files为True，将保留MinerU生成的JSON和其他文件
        """
        super().__init__()
        self.prefer_mineru = prefer_mineru
        self.save_intermediate_files = save_intermediate_files
        self.intermediate_output_dir = intermediate_output_dir
        
        # 检查依赖可用性
        if MINERU_AVAILABLE:
            self._log('MinerU 工具可用，将用于PDF智能解析')
        else:
            self._log_error('MinerU 工具不可用 - 表格解析将无法工作！')
            self._log_error('如果已安装MinerU但在此环境中不可用，请激活conda虚拟环境')
            self._log_error('或运行: pip install mineru')
            
        if PYPDF2_AVAILABLE:
            self._log('PyPDF2 工具可用，将用于PDF基础解析（仅提取文本，无法解析表格）')
        else:
            self._log_error('PyPDF2 工具不可用')
        
        if not MINERU_AVAILABLE and not PYPDF2_AVAILABLE:
            self._log_error('警告：没有可用的PDF解析工具！')
    
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表
        """
        return ['.pdf']
    
    def read_document(self, filepath):
        """
        【功能】从PDF文档中提取文字内容
        
        【参数】
            filepath: PDF文档的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 验证文件是否存在且可读
            2. 检查文件扩展名是否支持
            3. 优先尝试使用MinerU智能解析（如果可用且启用）
            4. 如果MinerU不可用或解析失败，回退到PyPDF2基础解析
            5. 将内容切分成适合检索的小块
            
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
        
        # 检查用户是否希望使用MinerU但MinerU不可用
        if self.prefer_mineru and not MINERU_AVAILABLE:
            self._log_error('警告：用户希望使用MinerU进行智能解析，但MinerU不可用！')
            self._log_error('表格解析将无法工作，只能提取基础文本。')
            self._log_error('解决方案：激活conda虚拟环境或安装MinerU: pip install mineru')
        
        # 优先尝试使用MinerU智能解析（如果可用且启用）
        if self.prefer_mineru and MINERU_AVAILABLE:
            try:
                self._log('尝试使用MinerU进行智能解析...')
                documents = self._read_pdf_with_mineru(filepath)
                self._log(f'MinerU智能解析成功，获得 {len(documents)} 个片段')
                return documents
            except Exception as e:
                self._log(f'MinerU解析失败，回退到PyPDF2基础解析: {e}')
                # 继续执行PyPDF2回退逻辑
        
        # MinerU不可用或解析失败，使用PyPDF2基础解析
        if PYPDF2_AVAILABLE:
            self._log('使用PyPDF2进行基础解析（仅提取文本，无法解析表格）')
            return self._read_pdf_with_pypdf2(filepath)
        else:
            self._log_error('没有可用的PDF解析工具！')
            return []
    
    def _read_pdf_with_mineru(self, filepath):
        """
        【功能】使用MinerU工具进行智能PDF解析
        
        【参数】
            filepath: PDF文档的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 使用MinerU的read_fn读取PDF二进制数据
            2. 调用do_parse进行智能解析，生成结构化JSON
            3. 读取JSON文件，提取文本和表格内容
            4. 使用智能内容分割器将内容切分成适合检索的小块
            5. 如果save_intermediate_files为True，保留所有中间文件
        """
        self._log(f'正在智能解析: {filepath}')
        
        try:
            # 读取PDF二进制数据
            pdf_bytes = read_fn(filepath)
            pdf_file_name = Path(filepath).stem
            
            # 确定输出目录
            if self.save_intermediate_files:
                # 使用永久目录保存中间文件
                if self.intermediate_output_dir:
                    output_dir = self.intermediate_output_dir
                else:
                    # 默认输出目录
                    output_dir = os.path.join(".", "mineru_output")

                # MinerU会在output_dir下创建一个以pdf_file_name为名的子目录
                # 所以实际输出结构是: output_dir/pdf_file_name/auto/...
                pdf_output_dir = os.path.join(output_dir, pdf_file_name)

                # 确保目录存在
                os.makedirs(pdf_output_dir, exist_ok=True)

                self._log(f'将保存中间文件到: {pdf_output_dir}')

                # 使用永久目录进行解析
                do_parse(
                    output_dir=output_dir,  # 注意：MinerU会在output_dir下创建子目录
                    pdf_file_names=[pdf_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],  # 默认使用中文
                    backend="pipeline"
                )

                # 查找生成的JSON文件（路径结构：output_dir/pdf_file_name/auto/...）
                content_file = os.path.join(pdf_output_dir, "auto", f"{pdf_file_name}_content_list.json")
                
                # 记录保存的文件
                self._log(f'中间文件已保存到: {pdf_output_dir}')
                self._log(f'JSON结果文件: {content_file}')
                
            else:
                # 使用临时目录（不保留中间文件）
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 调用MinerU的do_parse进行智能解析
                    # MinerU会在temp_dir下创建pdf_file_name子目录
                    do_parse(
                        output_dir=temp_dir,
                        pdf_file_names=[pdf_file_name],
                        pdf_bytes_list=[pdf_bytes],
                        p_lang_list=["ch"],  # 默认使用中文
                        backend="pipeline"
                    )

                    # 查找生成的JSON文件（路径结构：temp_dir/pdf_file_name/auto/...）
                    content_file = os.path.join(temp_dir, pdf_file_name, "auto", f"{pdf_file_name}_content_list.json")

                    # 检查文件是否存在（在with块内执行，此时temp_dir还存在）
                    if not os.path.exists(content_file):
                        self._log_error(f'未找到解析结果文件: {content_file}')
                        # 尝试列出auto目录内容以便调试
                        auto_dir = os.path.join(temp_dir, pdf_file_name, "auto")
                        if os.path.exists(auto_dir):
                            self._log_error(f'调试：auto_dir存在，内容: {os.listdir(auto_dir)}')
                        else:
                            self._log_error(f'调试：auto_dir不存在！')
                        raise FileNotFoundError(f"MinerU解析结果文件不存在: {content_file}")

                    # 读取JSON内容（在with块内读取，确保temp_dir还没被删除）
                    with open(content_file, 'r', encoding='utf-8') as f:
                        content_data = json.load(f)

                    self._log(f'成功解析PDF，获得 {len(content_data)} 个内容项')

                    # 统计表格和文本数量
                    text_count = sum(1 for item in content_data if item.get('type') == 'text')
                    table_count = sum(1 for item in content_data if item.get('type') == 'table')
                    self._log(f'解析结果统计: 文本项={text_count}, 表格项={table_count}')

                    self._log('开始智能分块处理...')

                    # 使用智能内容分割器（保持表格结构、标题级别等）
                    documents = self._split_mineru_content(content_data, chunk_size=300, overlap=30)

                    self._log(f'智能分块完成，获得 {len(documents)} 个片段')
                    return documents
                
        except Exception as e:
            self._log_error(f'解析文件 {filepath} 时出错: {e}')
            raise  # 重新抛出异常，让调用者处理
    
    def _read_pdf_with_pypdf2(self, filepath):
        """
        【功能】使用PyPDF2进行基础PDF解析
        
        【参数】
            filepath: PDF文档的文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文字（已经被切分）
            
        【处理流程】
            1. 使用PyPDF2打开PDF文件
            2. 遍历所有页面，提取文本内容
            3. 使用文本分割器将长文本切分成小块
        """
        self._log(f'正在基础解析: {filepath}')
        
        try:
            full_text = ''  # 存储完整文本
            
            # 使用PyPDF2打开PDF文件
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 遍历所有页面
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        full_text += page_text + '\n'  # 加换行符分隔页面
            
            self._log(f'原文共 {len(full_text)} 个字符')
            
            # 使用RecursiveCharacterTextSplitter切分文本
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # 块大小
                chunk_overlap=30  # 重叠大小
            )
            
            # split_text() 执行切分
            documents = splitter.split_text(full_text)
            
            self._log(f'切分成 {len(documents)} 个片段')
            
            return documents
            
        except Exception as e:
            self._log_error(f'读取文件 {filepath} 时出错: {e}')
            return []
    
    def _split_mineru_content(self, content_data, chunk_size=300, overlap=30):
        """
        【功能】智能分割MinerU解析的结构化内容
        
        【参数】
            content_data: MinerU解析的JSON数据列表
            chunk_size: 块大小（默认300字符）
            overlap: 重叠大小（默认30字符）
            
        【返回值】
            分割后的文本片段列表（纯文本）
            
        【处理逻辑】
            1. 识别文本项和表格项
            2. 根据标题级别(text_level)进行智能分块
            3. 表格内容单独处理，保持表格结构
            4. 返回适合检索的文本片段
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for item in content_data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if not text:
                    continue
                if item.get('text_level', 0) > 0:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [f"## {text}"]
                    current_length = len(text)
                elif current_length + len(text) > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [text]
                    current_length = len(text)
                else:
                    current_chunk.append(text)
                    current_length += len(text)

            elif item.get('type') == 'table':
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                table_body = item.get('table_body', '')
                if table_body:
                    if len(table_body) <= chunk_size:
                        chunks.append(table_body)
                    else:
                        table_chunks = self._split_mineru_table(table_body, chunk_size)
                        chunks.extend(table_chunks)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_mineru_table(self, table_html, chunk_size):
        """
        【功能】分割大型HTML表格
        
        【参数】
            table_html: 表格HTML内容
            chunk_size: 块大小
            
        【返回值】
            分割后的表格片段列表
        """
        chunks = []
        if table_html.startswith('<table>'):
            table_content = table_html[7:-8]
        else:
            table_content = table_html
        
        rows = table_content.split('<tr>')
        if len(rows) < 2:
            return [table_html]
        
        table_head = rows[1]
        current_chunk = '<table>' + table_head
        
        current_chunk_size = len(current_chunk)
        for row in rows[2:]:
            row_content = '<tr>' + row
            if current_chunk_size + len(row_content) > chunk_size and current_chunk != '<table>':
                current_chunk += '</table>'
                chunks.append(current_chunk)
                current_chunk = '<table>' + table_head + row_content
                current_chunk_size = len(current_chunk)
            else:
                current_chunk += row_content
                current_chunk_size += len(row_content)
        
        if current_chunk != '<table>' + table_head:
            current_chunk += '</table>'
            chunks.append(current_chunk)
        
        return chunks
    
    def get_parser_info(self):
        """
        【功能】获取解析器信息
        
        【返回值】
            解析器信息字典，包含可用工具和配置
        """
        info = {
            'prefer_mineru': self.prefer_mineru,
            'save_intermediate_files': self.save_intermediate_files,
            'intermediate_output_dir': self.intermediate_output_dir,
            'mineru_available': MINERU_AVAILABLE,
            'pypdf2_available': PYPDF2_AVAILABLE,
            'supported_extensions': self.get_supported_extensions(),
        }
        return info


# =============================================================================
# 模块测试代码
# =============================================================================
if __name__ == '__main__':
    """
    【功能】模块独立测试
    
    【使用方式】
        python pdf_document_reader.py
    """
    logger.info("=" * 100)
    logger.info("PDFDocumentReader 模块测试")
    logger.info("=" * 100)
    
    # 创建PDF文档读取器实例
    reader = PDFDocumentReader()
    
    # 测试支持的扩展名
    logger.info(f"支持的扩展名: {reader.get_supported_extensions()}")
    
    # 测试解析器信息
    info = reader.get_parser_info()
    logger.info(f"解析器信息:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # 测试文件类型检测
    test_files = [
        "test.pdf",       # 支持的文件
        "test.docx",      # 不支持的文件
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