# 基于Advance RAG的智能知识库系统

## 产品概述

《基于Advance RAG的智能知识库系统》是一款基于RAG（检索增强生成）技术的企业级智能知识库问答系统。该系统能够将企业文档转化为可智能检索的知识库，通过大语言模型为用户提供精准、高效的问答服务。系统采用模块化架构设计，支持多种文档格式，提供自动路由检索、混合检索、结果重排序等高级RAG功能。

---

## 核心功能

### 1. 智能文档管理
- **多格式文档上传**：支持 Word (.docx/.doc)、PDF、Markdown (.md)、纯文本 (.txt)、CSV、Excel (.xlsx) 等多种文档格式
- **自动分块处理**：采用智能文本分割技术，支持按Markdown标题结构（三级标题）智能分块、按字符数递归分割等多种策略
- **摘要索引**：自动为每个文档片段生成摘要，构建多向量检索器（MultiVectorRetriever），提升检索质量
- **向量化存储**：使用先进的Embedding技术将文档内容转化为向量，存储于ChromaDB向量数据库
- **文档管理**：支持查看文档列表、删除文档及其对应的向量数据

### 2. 智能问答系统
- **语义检索**：基于向量相似度检索，精准定位相关问题答案
- **混合检索**：支持向量检索 + BM25关键词检索的混合检索模式，可调节权重
- **结果重排序**：支持检索后重排序（Rerank），进一步提升结果相关性
- **自动路由检索**：支持两阶段路由检索——先通过路由引擎自动判断最相关的知识库，再在候选库中精准检索
- **上下文增强**：自动整合检索到的相关文档片段，构建丰富的上下文提示
- **大模型生成**：调用华言大语言模型，生成准确、自然的回答
- **检索溯源**：展示检索到的原始文档片段，支持查看引用来源

### 3. 多轮对话与历史管理
- **多轮对话**：支持基于知识库的多轮对话，自动整合历史对话上下文
- **对话历史持久化**：支持Redis存储（默认）和本地JSON文件存储（Redis不可用时自动降级）
- **历史消息管理**：自动保留最近10条消息作为上下文，支持查看历史对话

### 4. 用户认证与权限
- **工号登录**：通过员工工号进行身份认证
- **会话管理**：基于Cookie的会话管理，登录后自动跳转到聊天页面

### 5. Web交互界面
- **HarmonyOS风格UI**：现代化的毛玻璃（Glassmorphism）界面设计，提供流畅的用户体验
- **多轮对话**：支持连贯的上下文对话体验
- **文档管理页面**：便捷的上传、查看和管理企业文档
- **登录页面**：简洁的工号登录界面

### 6. 模型健康检测
- **实时监控**：支持检测对话模型、嵌入模型、重排序模型和PDF解析引擎的健康状态
- **轻量/完整检测**：支持轻量检测（初始化检查）和完整检测（实际调用测试）
- **单模型/全模型检测**：支持指定模型类型或检测所有模型

---

## 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层 (Web UI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   登录页面    │  │   聊天界面    │  │  文档上传页面   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (FastAPI App)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  main.py - Web服务入口，处理HTTP请求和路由               │   │
│  │  - 文档上传/删除/切换                                   │   │
│  │  - 智能问答                                           │   │
│  │  - 用户登录                                           │   │
│  │  - 模型健康检测                                        │   │
│  │  - 聊天历史查询                                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG服务层 (RAG Service)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  rag.py - RAGService类，核心业务逻辑编排                │   │
│  │  1. save_to_db: 文档处理 + 向量入库                     │   │
│  │  2. chat: 智能问答（路由 → 检索 → 重排序 → 生成）         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    服务层 (Service Layer)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  文档处理器   │  │  路由引擎    │  │  向量检索     │      │
│  │ document_   │  │ route_engine │  │  ChromaDB    │      │
│  │ processor.py│  │  .py         │  │              │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ 文档读取器族 │  │  LLM调用     │  │  工具函数库   │      │
│  │ document_   │  │  models.py   │  │ function_    │      │
│  │ reader_*.py │  │              │  │ tools.py     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      模型层 (Model Layer)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Embedding模型 │  │  大语言模型   │  │  重排序模型   │      │
│  │  Qwen3-      │  │  text-chat-  │  │  Qwen3-      │      │
│  │  Embedding-8B│  │  latest      │  │  Reranker-4B │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 核心技术栈

| 层级 | 技术组件 | 说明 |
|------|----------|------|
| Web框架 | FastAPI + Uvicorn | 高性能Python异步Web框架 |
| 模板引擎 | Jinja2 3.1.6 | 模板渲染引擎 |
| 向量数据库 | ChromaDB 1.3.5+ | 开源向量数据库，支持持久化存储 |
| 大语言模型 | 华言 text-chat-latest | 对话模型 |
| Embedding模型 | 华言 Qwen3-Embedding-8B | 文本向量化模型 |
| 重排序模型 | 华言 Qwen3-Reranker-4B | 检索结果重排序模型 |
| 文档处理 | python-docx、Docx2txtLoader、PyPDF2/MinerU、pandas、CSVLoader | 多格式文档解析 |
| 文本分割 | LangChain RecursiveCharacterTextSplitter | 递归字符文本分割器 |
| 路由引擎 | 自研路由（LLM/向量/混合三种策略） | 自动知识库路由选择 |
| 检索增强 | LangChain MultiVectorRetriever | 摘要索引 + 多向量检索 |
| 日志系统 | Python logging + RotatingFileHandler | 日志轮转与持久化 |

---

## 产品特性

### 1. 高精度语义检索
- 采用先进的Embedding技术，将文本转化为高维向量
- 支持Top-N相似度检索，精准定位相关内容
- 混合检索模式（向量+BM25关键词），权重可调
- 检索后重排序（Rerank），进一步提升结果质量

### 2. 智能路由检索
- 自动路由引擎，无需手动选择知识库
- 三种路由策略：LLM路由、向量路由、混合路由
- 混合路由加权融合

### 3. 多文档格式支持
- PDF（MinerU智能解析 + PyPDF2基础解析双引擎）
- Word（.docx / .doc）
- Excel（.xlsx，按行分块保留表头）
- CSV（逗号分隔值）
- Markdown / 纯文本（按标题结构智能分块）
- 自动文件类型检测，拒绝不支持的文件类型

### 4. 摘要索引增强
- 自动为每个文档片段生成摘要
- 构建多向量检索器，摘要+原始文档双重索引
- 检索时先匹配摘要，再召回原文

### 5. 数据持久化
- 向量数据本地持久化存储（./chroma目录）
- 支持多集合管理，便于分类存储不同文档
- 路由索引持久化（route_index.json）
- 对话历史本地JSON存储

### 6. 高可用设计
- Redis不可用时自动降级到本地文件存储
- PDF解析引擎自动降级（MinerU → PyPDF2）
- 混合检索失败时自动回退为向量检索
- 完整的错误日志和堆栈跟踪

---

## 使用场景

### 1. 企业知识管理
- 人力资源政策问答
- 财务制度查询
- 产品手册智能检索

### 2. 客服智能助手
- 常见问题自动回答
- 产品咨询智能回复
- 服务流程引导

### 3. 教育培训
- 课程资料智能检索
- 学习问题答疑
- 知识点快速定位

---

## 快速开始

### 环境要求
- Python 3.11+
- 华言API Key（用于访问大语言模型）

### 安装步骤

1. **安装依赖包**
```bash
pip install -r requirements.txt
```

2. **配置API Key**
```bash
# 设置环境变量
export HUAYAN_API_KEY=your_api_key_here
```

3. **启动服务**
```bash
uvicorn main:app
```

4. **访问系统**
打开浏览器访问：http://127.0.0.1:5000

### 使用流程

1. **登录系统**
   - 访问首页 `/login/` 进入登录页面
   - 输入工号进行身份认证

2. **上传文档**
   - 访问 `/document_upload/` 页面
   - 选择支持的文档格式（.docx、.pdf、.md、.txt、.xlsx、.csv）上传
   - 系统自动解析并建立向量索引

3. **开始问答**
   - 访问首页 `/chat/` 进入聊天界面
   - 输入问题，系统返回智能回答
   - 支持多轮对话，自动整合历史上下文

---

## 项目结构

```
├── main.py                       # FastAPI Web应用入口
├── rag.py                        # RAGService核心逻辑（路由/检索/重排序/生成）
├── function_tools.py             # 工具函数库（向量数据库、模型调用、重排序、健康检测）
├── models.py                     # 模型配置与客户端（华言LLM/Embedding/Rerank）
├── route_engine.py               # 路由引擎（LLM/向量/混合三种路由策略）
├── document_processor.py         # 文档处理类（多格式支持 + 摘要索引 + 多向量检索器）
├── document_reader_base.py       # 文档读取器抽象基类
├── document_reader_pdf.py        # PDF读取器（MinerU智能解析 + PyPDF2回退）
├── document_reader_word.py       # Word文档读取器（.docx / .doc）
├── document_reader_csv.py        # CSV文件读取器
├── document_reader_md.py         # Markdown/纯文本读取器（按标题分块）
├── document_reader_xlsx.py       # Excel文件读取器（按行分块保留表头）
├── logger.py                     # 日志系统（控制台 + 轮转文件）
├── requirements.txt              # 依赖包列表
├── Dockerfile                    # Docker容器化部署配置
├── route_index.json              # 路由索引持久化文件
├── chroma/                       # 向量数据库文件（ChromaDB持久化目录）
├── templates/                    # HTML模板
│   ├── login.html               # 登录页面
│   ├── chat.html                # 聊天界面（HarmonyOS风格UI）
│   └── document_upload.html     # 文档上传页面
├── uploads/                      # 上传文档存储目录
├── chat_history/                 # 对话历史JSON存储目录
├── logs/                         # 日志文件目录
│   └── app.log                  # 应用运行日志
├── static/                       # 静态文件
│   ├── favicon.ico               # 网站图标
│   ├── apple-touch-icon.png      # Apple设备图标
│   └── apple-touch-icon-precomposed.png
├── mineru_model/                 # MinerU PDF解析模型数据
└── mineru_output/                # MinerU PDF解析中间输出
```

---

## API接口说明

### 1. 用户登录
```
POST /login/
Content-Type: application/json

请求体:
{
    "employee_id": "工号"
}

返回:
{
    "message": "登录成功",
    "session_id": "工号"
}
备注: 自动设置Cookie（session_id），有效期1天
```

### 2. 文档上传
```
POST /document_upload/
Content-Type: multipart/form-data

参数:
- file: 文档文件（支持 .docx、.pdf、.md、.txt、.xlsx、.csv、.doc）

返回:
{
    "message": "文件上传成功",
    "filename": "文件名",
    "collection_name": "集合名称",
    "detail": "处理详情"
}
```

### 3. 智能问答
```
POST /chat/
Content-Type: application/json

请求体:
{
    "message": "用户问题",
    "top_k": 10,
    "hybrid_search": false,
    "vector_weight": 0.5,
    "rerank": false,
    "rerank_top_k": 5,
    "auto_route": false,
    "session_id": "用户工号"
}

返回:
{
    "response": "AI回答", 
    "search_results": ["检索到的相关文档片段列表"]
}
```

### 4. 获取/切换文档集合
```
GET /collection/
返回: {"name_list": ["文件名列表"], "collection_name": "当前集合名"}

POST /collection/
请求: {"collection_name": "集合名称"}
返回: {"message": "切换成功", "collection_name": "集合名称"}
```

### 5. 获取文档列表
```
GET /documents/
返回: {"message": "获取文档列表成功", "files": [{"name": "文件名", "size": 文件大小}]}
```

### 6. 删除文档
```
DELETE /document/?filename=文件名
返回: {"message": "文档删除成功", "filename": "文件名", "collection_name": "集合名称"}
备注: 同时删除向量数据库中的集合和本地文件
```

### 7. 获取聊天历史
```
GET /chat_history/?session_id=工号
返回: {"messages": [消息列表], "count": 消息数量}
```

### 8. 模型健康检测
```
GET /api/model-health?model_type=chat&test_type=light

参数:
- model_type: chat / embedding / rerank（可选，不指定则检测全部）
- test_type: light / full（可选，默认light）

返回:
{
    "status": "success",
    "data": {
        "chat": {"status": "online", "model": "text-chat-latest", ...},
        "embedding": {"status": "online", ...},
        "rerank": {"status": "online", ...},
        "mineru": {"status": "online", ...}
    }
}
```
