import json
import collections

for dataset_name in ["arxiv", "LAION1M", "ytb_audio", "tripclick"]:
    data_map = collections.defaultdict(set)

    file_path = f"/home/ec2-user/hybrid_hardness/semi-real/filterbenchmark/{dataset_name}/hardness_format"

    input_path = f"{file_path}/filters.json"

    with open(input_path, "r", encoding="utf-8") as f:
        filters = json.load(f)

    label = []
    for i in range(len(filters)):
        label.append(filters[i]["name"])
    
    print(label)

    output_path = f"{file_path}/payloads.jsonl"

    payloads = []

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            payloads.append(json.loads(line))

    new_line = [{} for _ in range(len(payloads))] 

    for line_no in range(len(payloads)):
        for key in label:
            if key in payloads[line_no]:
                new_line[line_no][key] = 1
    
    print(new_line[0])

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for d in new_line:              # d는 dict
                f.write(json.dumps(d))
                f.write("\n")
        print(f"✅ {output_path} 파일에 데이터가 성공적으로 저장되었습니다.")
    except IOError as e:
        print(f"❌ 파일 저장 중 오류가 발생했습니다: {e}")