"""Phase 2 探查:oddsportal 衍生盘(outright 夺冠/出线)可抓性。

为什么先探 outright:① 单页即全部球队赔率(无需逐场进 104 个详情页);② 直接对得上
我们锦标赛模拟的 P(Winner)/P(出线);③ outright 盘比赛 1X2 更低效,更可能有 edge。

探两个:
  - 2026(实时):看现在能否抓到夺冠赔率(用于「我方 P vs 市场」找当前分歧点)。
  - 2022(已结算,冠军=Argentina):若可抓 -> 能做真正的 outright 回测(ROI/CLV)。

判定点:是否遇 Cloudflare;DOM 能否提取 participant -> odds。

Run:  python -m worldcup2026.scrapers.probe_derivatives
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "op_profile"
SHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

TARGETS = [
    ("2022 夺冠", "https://www.oddsportal.com/football/world/world-cup-2022/outrights/"),
]

# outright 页通常是 participant 行 + 单列赔率;探查多种候选选择器
EXTRACT_JS = """() => {
  const cand = {};
  for (const sel of ["[data-testid='game-row']","div[class*='eventRow']",
                     "li[class*='flex']","div[class*='participant']",
                     "[data-testid*='outright']","a[href*='/football/']"]) {
    cand[sel] = document.querySelectorAll(sel).length;
  }
  // 尝试抓「名字 + 数字赔率」对:扫所有含一个 1.x~999 数字的短行
  const rows = [...document.querySelectorAll("div,li,tr")];
  const pairs = [];
  for (const r of rows) {
    const t = r.innerText.trim();
    if (t.length < 2 || t.length > 60 || t.split("\\n").length > 3) continue;
    const m = t.match(/^([A-Za-z .'-]{3,30})\\s+(\\d{1,3}\\.\\d{1,2})$/);
    if (m) pairs.push([m[1].trim(), parseFloat(m[2])]);
  }
  // 去重(同名取首个)
  const seen = new Set(); const uniq = [];
  for (const [n,o] of pairs){ if(!seen.has(n)){seen.add(n); uniq.push([n,o]);} }
  // 额外:dump flex 行的原始文本(名字/赔率可能分在不同子元素)
  const flexRows = [...document.querySelectorAll("li[class*='flex']")]
    .map(r => r.innerText.trim().replace(/\\n/g, " | "))
    .filter(t => t && t.length < 80).slice(0, 30);
  return {selectors: cand, pairs: uniq.slice(0, 40), flexRows};
}"""


def probe(page, label, url):
    print(f"\n{'='*60}\n{label}: {url}")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:  # noqa: BLE001
        print(f"  [goto 失败] {e}")
        return
    page.wait_for_timeout(5500)
    try:
        page.wait_for_load_state("networkidle", timeout=12000)
    except Exception:
        pass
    # 滚动触发懒加载,再等
    for _ in range(4):
        page.mouse.wheel(0, 1600)
        page.wait_for_timeout(900)
    page.wait_for_timeout(1500)

    title = page.title()
    body = page.inner_text("body")[:500]
    cf = ("Just a moment" in body or "Cloudflare" in body
          or "Checking your browser" in title or "challenge" in page.url)
    shot = SHOT_DIR / f"op_outright_{label.split()[0]}.png"
    try:
        page.screenshot(path=str(shot), full_page=False)
    except Exception:
        pass

    print(f"  URL终: {page.url}")
    print(f"  title: {title}")
    print(f"  cloudflare: {cf}")
    # 全文 + 全局「队名 赔率」正则(跨元素,容忍换行)
    info = page.evaluate("""() => {
      const body = document.body.innerText;
      // 找 「单词串(<=24字符) 紧跟 1.01~999.0 赔率」,跨空白/换行
      const re = /([A-Z][A-Za-z .'&-]{2,24})\\s*\\n?\\s*(\\d{1,3}\\.\\d{2})\\b/g;
      const pairs = []; let m;
      while ((m = re.exec(body)) && pairs.length < 50)
        pairs.push([m[1].trim(), parseFloat(m[2])]);
      return {len: body.length, pairs,
              slice: body.slice(1200, 3200)};
    }""")
    print(f"  body 长度: {info['len']}")
    print(f"  全局正则 队名+赔率 命中: {len(info['pairs'])}")
    for n, o in info["pairs"][:20]:
        print(f"    {n:<26s} {o}")
    print(f"  ---- body[1200:3200] ----\n{info['slice']}")
    print(f"  截图: {shot}")


def main(headless: bool = True):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR), headless=headless, user_agent=UA,
            viewport={"width": 1366, "height": 900}, locale="en-US",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        for label, url in TARGETS:
            probe(page, label, url)
        ctx.close()


if __name__ == "__main__":
    main()
