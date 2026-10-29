#!/bin/sh
# ablation.py = interp_bench_v2.py sans son main + ablation_body.py (colab run n'envoie qu'un fichier)
cd "$(dirname "$0")" && { sed '/^if __name__/,$d' interp_bench_v2.py; cat ablation_body.py; } > ablation.py
