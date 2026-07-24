# ============================================
# MAZU 沙特多灾种预警智能体 — Docker 镜像
# ============================================

FROM python:3.10-slim AS builder

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先安装 Python 依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# 运行阶段
# ============================================
FROM python:3.10-slim

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r mazu && useradd -r -g mazu -d /app -s /sbin/nologin mazu

WORKDIR /app

# 从构建阶段复制 Python 包
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:$PATH"

# 复制应用代码
COPY . .

# 安装为可编辑包（注册 mazu-web / mazu-cli 命令）
RUN pip install -e .

# 创建运行时目录并设置权限
RUN mkdir -p outputs/models outputs/logs outputs/reports \
    && chown -R mazu:mazu /app

# 切换到非 root 用户
USER mazu

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:7860/ || exit 1

EXPOSE 7860

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    MAZU_PRODUCTION=1

CMD ["python", "run.py", "web", "--port", "7860"]
