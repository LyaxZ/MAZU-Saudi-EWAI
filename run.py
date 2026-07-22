"""MAZU 沙特多灾种预警智能体 — 统一入口"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="MAZU 沙特多灾种预警智能体")
    sub = parser.add_subparsers(dest="command", help="运行模式")

    # web: 启动 Gradio Web 界面
    web = sub.add_parser("web", help="启动 Gradio Web 对话界面")
    web.add_argument("--port", type=int, default=7860, help="服务端口 (默认: 7860)")
    web.add_argument("--share", action="store_true", help="生成公网分享链接")

    # cli: 命令行对话
    cli = sub.add_parser("cli", help="命令行对话模式")
    cli.add_argument("--model", type=str, default="deepseek-v4-flash",
                     help="模型名称 (默认: deepseek-v4-flash)")
    cli.add_argument("--verbose", action="store_true", help="显示调试信息")

    # train: 训练所有模型
    sub.add_parser("train", help="训练并保存四灾害 LightGBM 模型")

    # tools: 生成可视化产物
    tools = sub.add_parser("tools", help="生成可视化产物")
    tools.add_argument("target", choices=["kg", "events", "features", "comparison", "all"],
                       help="生成目标: kg/events/features/comparison/all")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "web":
        from app.gradio_app import main as web_main
        web_main()

    elif args.command == "cli":
        from app.chat_cli import main as cli_main
        sys.argv = [sys.argv[0]]
        if args.verbose:
            sys.argv.append("--verbose")
        cli_main()

    elif args.command == "train":
        from models.inference import DisasterInference
        infer = DisasterInference()
        infer.train_all()
        print("✅ 四灾害模型训练完成")

    elif args.command == "tools":
        if args.target in ("kg", "all"):
            import tools.generate_kg_html
        if args.target in ("events", "all"):
            import tools.generate_kg_events
        if args.target in ("features", "all"):
            import tools.plot_feature_importance
        if args.target in ("comparison", "all"):
            import tools.plot_model_comparison


if __name__ == "__main__":
    main()
