# 使用Python 3.11作为基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 定义环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 运行默认命令
CMD ["bash"]
