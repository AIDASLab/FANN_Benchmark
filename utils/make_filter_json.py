import json
import collections

for dataset_name in ["sift_high", "sift_low", "gist_high", "gist_low"]:

    data_map = collections.defaultdict(set)

    file_path = f"/home/ec2-user/hybrid_hardness/Benchmark/{dataset_name}"

    input_path = f"{file_path}/tests.jsonl"

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {line_no}번째 줄 JSON 파싱 실패: {e}")
                continue

            and_part = item.get("conditions", {}).get("and", [])
            if line_no == 1:
                print(and_part)
            if not and_part:
                continue

            for dic_part in and_part:
                if not isinstance(dic_part, dict):
                    continue

                # 첫 번째 키/값
                dic_key, nested_dict = next(iter(dic_part.items()))
                match_value = (nested_dict or {}).get("match", {}).get("value", None)

                if dic_key is not None and match_value is not None:
                    # match_value가 리스트일 수도 있으니 처리
                    if isinstance(match_value, list):
                        data_map[dic_key].update(v for v in match_value if v is not None and v != "")
                    else:
                        data_map[dic_key].add(match_value)

    # 키 목록
    name = list(data_map.keys())


    # 혹시 남아있는 None 제거(이중 안전장치)
    for s in data_map.values():
        if None in s:
            s.discard(None)
        # 빈 문자열도 제거하고 싶다면:
        if "" in s:
            s.discard("")

    # 최종 출력
    final_list = []
    for key, values_set in data_map.items():
        entry = {
            "name": key,
            "values": sorted(list(values_set))
        }
        final_list.append(entry)

    print(final_list)

    output_path = f"{file_path}/filters.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=2)
        print(f"✅ {output_path} 파일에 데이터가 성공적으로 저장되었습니다.")
    except IOError as e:
        print(f"❌ 파일 저장 중 오류가 발생했습니다: {e}")
