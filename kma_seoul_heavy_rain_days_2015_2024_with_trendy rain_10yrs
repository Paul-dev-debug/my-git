# 이 파일은 data.go.kr에서 서울 10년 강수량, 추세선을 구하는 python code입니다. anaconda prompt에서 구동되며, 11열의 {insert decoding code here}을 지우고 발급받으신 decoding code(API키)를 입력해 주세요.

# kma_seoul_heavy_rain_days_2015_2024_with_trend.py
# 실행:  python kma_seoul_heavy_rain_days_2015_2024_with_trend.py
# ───────────────────────────────────────────────────────────────
import requests, pandas as pd, numpy as np, matplotlib.pyplot as plt
from io import StringIO
import xml.etree.ElementTree as ET

SERVICE_KEY = (
    "{insert decoding code here}"
)

BASE_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
NUM_ROWS = 365
PARAM_BASE = {
    "serviceKey": SERVICE_KEY,
    "dataCd"   : "ASOS",
    "dateCd"   : "DAY",
    "stnIds"   : "108",          # 서울
    "dataType" : "XML",
    "numOfRows": NUM_ROWS,
}

YEARS  = range(2015, 2025)       # 2015‒2024
THRESH = 50.0                    # 폭우 기준(mm)

heavy_counts = {}

for yr in YEARS:
    start_dt, end_dt = f"{yr}0101", f"{yr}1231"
    print(f"\n▶ {start_dt} ~ {end_dt}")

    params = PARAM_BASE | {"startDt": start_dt, "endDt": end_dt, "pageNo": 1}
    resp   = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    if root.findtext(".//resultCode") != "00":
        raise RuntimeError(
            f"KMA API Error {root.findtext('.//resultCode')}: "
            f"{root.findtext('.//resultMsg')}"
        )

    df_year = pd.read_xml(StringIO(resp.text), xpath=".//item")
    if df_year is None or df_year.empty:
        heavy_counts[yr] = 0
        continue

    df_year["sumRn"] = pd.to_numeric(df_year["sumRn"], errors="coerce")
    heavy_counts[yr]  = int((df_year["sumRn"] >= THRESH).sum())
    print(f"  heavy rain days ≥{THRESH} mm : {heavy_counts[yr]}")

# ── Series & 저장 ────────────────────────────────────────────
heavy_ser = pd.Series(heavy_counts).sort_index()
heavy_ser.to_csv("seoul_heavy_rain_days_2015_2024.csv",
                 header=["days"], index_label="year")

# ── 선형 추세선 ──────────────────────────────────────────────
years  = heavy_ser.index.values.astype(int)
counts = heavy_ser.values.astype(float)
coef   = np.polyfit(years, counts, 1)
trend  = np.poly1d(coef)(years)

ss_res = ((counts - trend) ** 2).sum()
ss_tot = ((counts - counts.mean()) ** 2).sum()
r2     = 1 - ss_res / ss_tot

# ── 그래프 ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.bar(years, counts, label="Heavy-rain days (≥50 mm)")
plt.plot(years, trend, color="orange", linewidth=2, label="Linear trend")

for x, y in zip(years, counts):
    plt.text(x, y + 0.25, f"{int(y)}", ha="center", va="bottom", fontsize=8)

plt.title("Number of Heavy-Rain Days (≥50 mm) – Seoul (ID 108, 2015-2024)")
plt.xlabel("Year")
plt.ylabel("days")

# 범례를 오른쪽 위로 이동
plt.legend(loc="upper right")

# 회귀식 & R² 박스: 왼쪽 위에서 살짝 내려 배치
eq = f"y = {coef[0]:.2f}x + {coef[1]:.1f}\n$R^2$ = {r2:.3f}"
plt.text(0.02, 0.88, eq, transform=plt.gca().transAxes,
         va="top", bbox=dict(boxstyle="round", alpha=0.3, pad=0.4))

plt.tight_layout()
plt.show()
