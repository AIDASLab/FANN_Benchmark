import json
import matplotlib.pyplot as plt

# 1. JSON 불러오기
json_path = "/home/mintaek/hybrid_index/methods/ACORN/mw_practice/acorn_trade_off_1.json"
with open(json_path, "r") as f:
    ACORN_trade_off = json.load(f)


def plot_prefix(trade_off, prefix, title, out_path):
    """
    prefix: "gen_3" or "gen_12"
    lines:  prefix + "_high", "_low", "_random"
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # suffix마다 색/마커 지정 (보내준 그림 스타일)
    cfg = {
        "high":   {"color": "orange", "marker": "D"},  # 주황 다이아
        "low":    {"color": "green",  "marker": "D"},  # 초록 다이아
        "random": {"color": "black",  "marker": "D"},  # 검정 다이아
    }

    # 그릴 순서
    suffixes = ["high", "low", "random"]

    for suf in suffixes:
        key = f"{prefix}_{suf}"          # e.g. "gen_12_low"
        param_dict = trade_off[key]      # dict: param_str -> {"qps":..., "avg_recall":...}

        # QPS 기준으로 정렬해서 곡선이 자연스럽게
        items = sorted(param_dict.items(), key=lambda kv: kv[1]["qps"])
        qps_vals    = [v["qps"] for _, v in items]
        recall_vals = [v["avg_recall"] for _, v in items]

        style = cfg[suf]

        ax.plot(
            qps_vals,
            recall_vals,
            marker=style["marker"],
            linestyle="-",
            linewidth=2.5,
            markersize=8,
            color=style["color"],
            label=key.replace("gen_", ""),  # legend 라벨: "12_high" 등
        )

    # 축/타이틀/그리드/폰트 크기 살짝 키우기
    ax.set_xlabel("QPS", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="-", alpha=0.3)

    # legend를 아래쪽 중앙에 배치 (보내준 그림처럼)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved: {out_path}")


# 2. gen_3 그래프
plot_prefix(
    ACORN_trade_off,
    prefix="gen_1",
    title="QPS vs Recall Trade-off (18 datasets)",
    out_path="/home/mintaek/hybrid_index/methods/ACORN/mw_practice/trade_off_gen1_lines.png",
)

