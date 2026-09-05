"""CPT 实验的数据准备：下载 → 文章级划分 → tokenize → litdata chunk → 检查。

从仓库的 con-pretrain/ 目录下以模块方式运行，例如：
    cd con-pretrain && python -m data_prep.download --dataset biomed --dry-run
"""
