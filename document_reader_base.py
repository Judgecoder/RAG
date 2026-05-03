"""
================================================================================
文档读取器基类模块 (document_reader_base.py)

【这个文件的作用】
定义文档读取器的统一接口和基类，支持多态和扩展

【设计理念】
1. 抽象基类：定义统一的文档读取接口
2. 类型安全：通过ABC确保子类实现必要方法
3. 易于扩展：添加新文档类型只需继承基类
4. 统一错误处理：基类提供标准的异常处理机制
================================================================================
"""

from abc import ABC, abstractmethod
import os

from logger import logger


class DocumentReaderBase(ABC):
    """
    【类名】DocumentReaderBase
    
    【作用】文档读取器的抽象基类，定义统一接口
    
    【设计理念】
    1. 抽象接口：强制子类实现read_document方法
    2. 通用功能：提供文件类型检测、错误处理等基础功能
    3. 扩展友好：子类可以专注于特定格式的解析逻辑
    
    【子类要求】
    1. 必须实现read_document()方法
    2. 可以重写supports_extension()方法以支持更多扩展名
    3. 应该使用基类提供的日志和错误处理功能
    """
    
    @abstractmethod
    def read_document(self, filepath):
        """
        【功能】读取文档内容（抽象方法）
        
        【参数】
            filepath: 文档文件路径
            
        【返回值】
            文档内容列表，每个元素是一段文本（已切分）
            
        【异常】
            子类必须实现此方法，否则会抛出TypeError
        """
        pass
    
    def supports_extension(self, extension):
        """
        【功能】检查是否支持指定文件扩展名
        
        【参数】
            extension: 文件扩展名（小写，包含点，如'.docx'）
            
        【返回值】
            bool: 如果支持该扩展名则返回True
            
        【默认实现】
            子类应该重写此方法，返回支持的扩展名列表
        """
        return extension in self.get_supported_extensions()
    
    @abstractmethod
    def get_supported_extensions(self):
        """
        【功能】获取支持的扩展名列表
        
        【返回值】
            支持的扩展名列表，如['.docx', '.doc']
            
        【说明】
            子类必须实现此方法，返回该解析器支持的扩展名
        """
        pass
    
    def _log(self, message):
        """
        【功能】通用日志记录方法
        
        【参数】
            message: 日志消息
            
        【说明】
            子类可以使用此方法进行统一的日志输出
        """
        class_name = self.__class__.__name__
        logger.info(f'【{class_name}】{message}')
    
    def _log_error(self, message):
        """
        【功能】通用错误日志记录方法
        
        【参数】
            message: 错误消息
            
        【说明】
            子类可以使用此方法进行统一的错误日志输出
        """
        class_name = self.__class__.__name__
        logger.error(f'【{class_name}错误】{message}')
    
    def _validate_file(self, filepath):
        """
        【功能】验证文件是否存在且可读
        
        【参数】
            filepath: 文件路径
            
        【返回值】
            bool: 文件有效则返回True，否则返回False
            
        【异常】
            如果文件不存在或不可读，记录错误并返回False
        """
        if not os.path.exists(filepath):
            self._log_error(f'文件不存在: {filepath}')
            return False
        
        if not os.path.isfile(filepath):
            self._log_error(f'不是有效文件: {filepath}')
            return False
        
        if not os.access(filepath, os.R_OK):
            self._log_error(f'文件不可读: {filepath}')
            return False
        
        return True
    
    def _get_file_extension(self, filepath):
        """
        【功能】获取文件扩展名（小写）
        
        【参数】
            filepath: 文件路径
            
        【返回值】
            文件扩展名（小写，包含点），如'.docx'
        """
        _, extension = os.path.splitext(filepath)
        return extension.lower()