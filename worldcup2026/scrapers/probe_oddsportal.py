"""探查 oddsportal 2022 世界杯结果页结构(一次性):

  - 是否遇 Cloudflare 挑战
  - 赔率走哪个 XHR feed(端点 URL 形态)
  - 渲染后 DOM 能否直接提取比赛+赔率

Run:  python -m worldcup2026.scrapers.probe_oddsportal
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://www.oddsportal.com/football/world/world-cup-2022/results/"
PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "op_profile"
SHOT = Path(__file__).resolve().parent.parent / "data" / "cache" / "op_probe.png"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")


def main(headless: bool = True):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    feeds = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            user_agent=UA,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")

        archive_body = {"text": None}

        def on_resp(r):
            u = r.url
            if any(k in u for k in ("/feed/", "ajax", ".json", "/api/", "match-event")):
                feeds.append((r.status, u[:160]))
            if "tournament-archive" in u:
                try:
                    archive_body["text"] = r.text()
                except Exception as e:  # noqa: BLE001
                    archive_body["text"] = f"<read err {e}>"

        page.on("response", on_resp)
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(6000)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        title = page.title()
        body = page.inner_text("body")[:600]
        page.screenshot(path=str(SHOT), full_page=False)

        # --- DOM 行结构探查 ---
        print("\n=== DOM 选择器命中 ===")
        for sel in ["div[class*='eventRow']", "[data-testid='game-row']",
                    "div[set]", "[class*='participant']", "[data-testid*='odd']"]:
            print(f"  {sel:35s} -> {len(page.query_selector_all(sel))}")
        print("\n=== game-row 结构(取含比分的前 4 行)===")
        data = page.evaluate("""() => {
          const rows=[...document.querySelectorAll("[data-testid='game-row']")];
          const out=[];
          for(const r of rows){
            const parts=[...r.querySelectorAll("[class*='participant']")].map(e=>e.innerText.trim()).filter(Boolean);
            const odds=[...r.querySelectorAll("[data-testid*='odd']")].map(e=>e.innerText.trim()).filter(Boolean);
            const link=r.querySelector("a[href*='/football/']");
            out.push({parts, odds, href: link?link.getAttribute('href'):null,
                      text: r.innerText.replace(/\\n/g,' | ')});
          }
          return out;
        }""")
        print(f"game-row 总数: {len(data)}")
        shown = 0
        for d in data:
            if len(d["parts"]) >= 2 and len(d["odds"]) >= 3:
                print(f"\nparts={d['parts']}")
                print(f"odds={d['odds']}")
                print(f"href={d['href']}")
                print(f"text={d['text'][:160]}")
                shown += 1
                if shown >= 4:
                    break

        # 检测 Cloudflare
        cf = ("Just a moment" in body or "Cloudflare" in body
              or "Checking your browser" in title or "challenge" in page.url)
        # 统计页面里疑似比赛行
        rows = page.query_selector_all("div[class*='eventRow'], a[href*='/football/']")

        print(f"URL       : {page.url}")
        print(f"title     : {title}")
        print(f"cloudflare: {cf}")
        print(f"事件行数  : {len(rows)}")
        print(f"feed/xhr 命中 ({len(feeds)}):")
        for s, u in feeds[:25]:
            print(f"   [{s}] {u}")
        print(f"\nbody 预览 :\n{body[:400]}")
        print(f"\n截图: {SHOT}")
        ctx.close()


if __name__ == "__main__":
    main()
