from math import isnan
import numpy as np

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def extract_points(stats_dict):
    """
    stats_dict: {key: {'qps': float, 'avg_recall': float, ...}, ...}
    반환: [(qps, recall), ...]  (유효한 값만)
    """
    pts = []
    for _, v in stats_dict.items():
        # 키 이름 변형 가능성 대비
        qps = v.get('qps', v.get('QPS'))
        rec = v.get('avg_recall', v.get('recall', v.get('AvgRecall')))
        qps = _safe_float(qps)
        rec = _safe_float(rec)
        if qps is None or rec is None:
            continue
        if isnan(qps) or isnan(rec):
            continue
        pts.append((qps, rec))
    return pts

def pareto_front(points):
    """
    points: [(qps, recall), ...]
    QPS와 Recall을 모두 '큰 값이 좋다'고 가정.
    반환: Pareto front 위의 점들을 qps 오름차순으로 정렬한 리스트.
    """
    if not points:
        return []
    # (qps 내림차순, recall 내림차순)으로 정렬 후 스윕
    uniq = sorted({(float(q), float(r)) for q, r in points},
                  key=lambda x: (-x[0], -x[1]))
    front, best_r = [], -1.0
    for q, r in uniq:
        if r > best_r:
            front.append((q, r))
            best_r = r
    # 보기 편하게 qps 오름차순으로 정렬
    front.sort(key=lambda x: x[0])
    return front

def mean_of_front(stats_dict):
    """
    전체 점들에서 Pareto front를 구한 뒤 front 위 점들의 평균 (mean_qps, mean_recall) 반환.
    front가 비면 (0.0, 0.0) 반환.
    """
    pts = extract_points(stats_dict)
    front = pareto_front(pts)
    if not front:
        return (0.0, 0.0), []
    mq = sum(q for q, _ in front) / len(front)
    mr = sum(r for _, r in front) / len(front)
    return (mq, mr), front

def final_score (stats_dict):
    (mq, mr), front_pts = mean_of_front(stats_dict)
    return np.log10(mq) * mr