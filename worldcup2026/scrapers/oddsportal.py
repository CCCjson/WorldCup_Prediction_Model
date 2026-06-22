"""oddsportal 国家队赔率抓取(playwright + 持久 profile + 反检测)。

只取每场的平均 1X2 收盘赔率(o1/ox/o2);**不取 oddsportal 的比分**(淘汰赛会
混入点球/加时,口径乱),比赛结果(outcome)一律用本项目历史数据集的比分对齐。

archive feed 是 base64+加密(不可逆向),因此从渲染后的 DOM 明文提取。

Run:  python -m worldcup2026.scrapers.oddsportal          # 抓 2022 世界杯并缓存
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
PROFILE_DIR = CACHE_DIR / "op_profile"
ALIAS_PATH = Path(__file__).resolve().parent.parent / "data" / "alias_map.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

EXTRACT_JS = """() => {
  const rows=[...document.querySelectorAll("[data-testid='game-row']")];
  const out=[];
  for(const r of rows){
    const parts=[...r.querySelectorAll("[class*='participant']")].map(e=>e.innerText.trim()).filter(Boolean);
    const odds=[...r.querySelectorAll("[data-testid*='odd']")].map(e=>e.innerText.trim()).filter(Boolean);
    const status=r.innerText.split("\\n")[0].trim();
    if(parts.length>=2 && odds.length>=5){
      // odds 每个值重复两次 -> 取 0,2,4
      out.push({home:parts[0], away:parts[1],
                o1:odds[0], ox:odds[2], o2:odds[4], status});
    }
  }
  return out;
}"""

FINISHED = ("Finished", "After Pen.", "After ET", "awarded")


def _load_alias():
    raw = json.loads(ALIAS_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def scrape_tournament(url: str, cache_name: str, valid_teams=None, expected=64,
                      max_pages: int = 8, headless: bool = True,
                      force: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / cache_name
    if cache.exists() and not force:
        print(f"[cache] {cache.name} present — loading")
        return pd.read_parquet(cache)

    alias = _load_alias()
    valid = set(valid_teams) if valid_teams else None
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), headless=headless, user_agent=UA,
            viewport={"width": 1366, "height": 900}, locale="en-US",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4500)

        def harvest():
            data = page.evaluate(EXTRACT_JS)
            new = 0
            for d in data:
                if not any(d["status"].startswith(f) for f in FINISHED):
                    continue
                h = alias.get(d["home"], d["home"])
                a = alias.get(d["away"], d["away"])
                if valid and (h not in valid or a not in valid):
                    continue
                key = frozenset((h, a))
                if key in seen:
                    continue
                try:
                    o1, ox, o2 = float(d["o1"]), float(d["ox"]), float(d["o2"])
                except ValueError:
                    continue
                seen.add(key)
                rows.append({"home_op": d["home"], "away_op": d["away"],
                             "home_team": h, "away_team": a,
                             "odds_home": o1, "odds_draw": ox, "odds_away": o2})
                new += 1
            return new

        for pg in range(1, max_pages + 1):
            if pg > 1:
                clicked = page.evaluate(
                    """(n)=>{const ls=[...document.querySelectorAll('a.pagination-link')];
                    const t=ls.find(l=>l.innerText.trim()===String(n));
                    if(t){t.click();return true;}return false;}""", pg)
                if not clicked:
                    break
                page.wait_for_timeout(4000)
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
            new = harvest()
            print(f"  page {pg}: +{new} 场(累计 {len(rows)})")
            if len(rows) >= expected:
                break
            if new == 0 and pg > 1:
                break
        ctx.close()

    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    print(f"[write] {cache} ({len(df)} 场)")
    return df


WC2022_URL = "https://www.oddsportal.com/football/world/world-cup-2022/results/"


if __name__ == "__main__":
    from worldcup2026.sim.tournament import config_2022
    teams = [t for g in config_2022()["groups"].values() for t in g]
    df = scrape_tournament(WC2022_URL, "oddsportal_wc2022.parquet",
                           valid_teams=teams, expected=64, force=True)
    print(f"\n抓到 {len(df)} 场(2022 WC 应为 64)")
    print(df[["home_team", "away_team", "odds_home", "odds_draw", "odds_away"]]
          .head(10).to_string(index=False))
