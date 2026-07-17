"""测试 MAZU-FengYun API 返回的数据详情"""
import requests, json

base = "https://fyearth.nsmc.org.cn/fydq/v2/api"

tests = [
    ("OVW 原始风场", "OVW", "Global", "02010102"),
    ("OVW-FILL 风速填色", "OVW-FILL", "Global", "02010103"),
    ("SST 日平均", "SST", "Global", "02060001"),
    ("MRR 降水", "MRR", "Global", "02030300"),
    ("NVI-AVE NDVI月均", "NVI-AVE", "Global", "02050100"),
    ("OLR 长波辐射", "OLR", "Global", "02070100"),
]

for name, abbr, region, code in tests:
    print(f"\n{'='*60}")
    print(f"  {name} ({abbr})")
    r = requests.get(f"{base}/content", params={
        "abbr": abbr, "region": region, "code": code, "language": "EN"
    })
    if r.status_code != 200:
        print(f"  Status: {r.status_code}")
        continue
    data = r.json()
    if not data:
        print("  无数据")
        continue
    item = data[0]["list"][0]
    print(f"  卫星: {item.get('strSatID')}/{item.get('strInstrument')}")
    print(f"  分辨率: {item.get('resolution')}°")
    print(f"  尺寸: {item.get('width')}x{item.get('height')}")
    print(f"  时间: {item.get('strBTime')} ~ {item.get('strETime')}")
    print(f"  数据类型: {item.get('dataType')}")
    # 检查所有字段
    for k, v in item.items():
        if k.endswith("FileName") and v:
            print(f"  {k}: {v[:120]}...")

# 尝试下载一张图片看看格式
print(f"\n{'='*60}")
print("  下载 OVW 原始风场图片...")
r = requests.get(f"{base}/content", params={
    "abbr": "OVW", "region": "Global", "code": "02010102", "language": "EN"
})
if r.status_code == 200:
    data = r.json()
    if data:
        img_url = data[0]["list"][0]["oriImgFileName"]
        r2 = requests.get(img_url, timeout=30)
        print(f"  Content-Type: {r2.headers.get('Content-Type')}")
        print(f"  文件大小: {len(r2.content):,} bytes")
        # 检查是否是真正的PNG
        if r2.content[:4] == b'\x89PNG':
            print(f"  格式: PNG (已验证)")
        else:
            print(f"  前4字节: {r2.content[:4].hex()}")
