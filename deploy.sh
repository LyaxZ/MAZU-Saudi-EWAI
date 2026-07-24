#!/bin/bash
# ============================================================
# MAZU 沙特多灾种预警智能体 — 服务器一键部署脚本
# 适用: Ubuntu 24.04 + Miniconda3 (root)
# 用法: bash deploy.sh
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { echo -e "${GREEN}[MAZU]${NC} $1"; }
warn() { echo -e "${RED}[WARN]${NC} $1"; }

# ── 0. 检测 ──────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   MAZU 沙特多灾种预警智能体 — 服务器部署  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
log "项目目录: $APP_DIR"

# 检查 conda
if ! command -v conda &>/dev/null; then
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        source /root/miniconda3/etc/profile.d/conda.sh
    else
        warn "未找到 conda，请先安装 Miniconda3"; exit 1
    fi
fi
log "conda: $(conda --version)"

# ── 1. 创建 conda 环境 ─────────────────────────────
ENV_NAME="mazu"
if conda env list | grep -q "^${ENV_NAME}\s"; then
    log "conda 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    log "创建 conda 环境 '${ENV_NAME}' (Python 3.10)..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# ── 2. 安装依赖 ─────────────────────────────────────
log "安装 Python 依赖..."
pip install --upgrade pip -q
pip install -r "${APP_DIR}/requirements.txt" -q
pip install -e "${APP_DIR}" -q
log "依赖安装完成"

# ── 3. 检查 .env ────────────────────────────────────
if [ ! -f "${APP_DIR}/.env" ]; then
    if [ -f "${APP_DIR}/.env.example" ]; then
        warn ".env 不存在，从 .env.example 复制..."
        cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
        echo ""
        echo -e "${RED}⚠️  请编辑 ${APP_DIR}/.env 填入真实的 API Key！${NC}"
        echo -e "${RED}   nano ${APP_DIR}/.env${NC}"
        echo ""
    else
        warn ".env.example 也不存在，请手动创建 .env"
    fi
else
    log ".env 已存在"
fi

# ── 4. 创建运行时目录 ─────────────────────────────
mkdir -p "${APP_DIR}/outputs/models"
mkdir -p "${APP_DIR}/outputs/logs"
mkdir -p "${APP_DIR}/outputs/reports"
log "运行时目录已创建"

# ── 5. 检查数据目录 ─────────────────────────────────
if [ ! -d "${APP_DIR}/indicators" ]; then
    warn "indicators/ 目录不存在 — 预测功能不可用"
    warn "请将 NC 数据文件放入 ${APP_DIR}/indicators/"
fi

# ── 6. 检查模型文件 ─────────────────────────────────
MODEL_COUNT=$(ls -1 "${APP_DIR}/outputs/models/"*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 4 ]; then
    warn "模型文件不完整 (${MODEL_COUNT}/4)，首次启动时将自动训练"
fi

# ── 7. 安装 systemd 服务 ───────────────────────────
SERVICE_FILE="/etc/systemd/system/mazu.service"
cat > /tmp/mazu.service << SERVEOF
[Unit]
Description=MAZU Saudi Multi-Hazard Early Warning AI Agent
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
Environment="PATH=/root/miniconda3/envs/${ENV_NAME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="MAZU_PRODUCTION=1"
Environment="MAZU_CONCURRENCY=5"
EnvironmentFile=-${APP_DIR}/.env
ExecStart=/root/miniconda3/envs/${ENV_NAME}/bin/python ${APP_DIR}/run.py web --port 7860
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/outputs/logs/mazu_stdout.log
StandardError=append:${APP_DIR}/outputs/logs/mazu_stderr.log

[Install]
WantedBy=multi-user.target
SERVEOF

if [ -f "$SERVICE_FILE" ]; then
    log "systemd 服务文件已存在，更新..."
fi
cp /tmp/mazu.service "$SERVICE_FILE"
systemctl daemon-reload
log "systemd 服务已安装: mazu.service"

# ── 8. 开放防火墙端口 ─────────────────────────────
if command -v ufw &>/dev/null; then
    ufw allow 7860/tcp 2>/dev/null && log "防火墙: 已开放 7860 端口" || log "防火墙: 7860 可能已开放"
else
    log "未检测到 ufw，请手动确认 7860 端口已开放"
fi

# ── 9. 完成 ─────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         部署完成！接下来请执行：         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  1. ${CYAN}编辑 .env${NC} (如果还没填):"
echo -e "     nano ${APP_DIR}/.env"
echo ""
echo -e "  2. ${CYAN}启动服务${NC}:"
echo -e "     systemctl start mazu"
echo ""
echo -e "  3. ${CYAN}设置开机自启${NC}:"
echo -e "     systemctl enable mazu"
echo ""
echo -e "  4. ${CYAN}查看状态${NC}:"
echo -e "     systemctl status mazu"
echo ""
echo -e "  5. ${CYAN}查看日志${NC}:"
echo -e "     journalctl -u mazu -f"
echo ""
echo -e "  6. ${CYAN}访问${NC}:"
echo -e "     http://<服务器IP>:7860"
echo ""
