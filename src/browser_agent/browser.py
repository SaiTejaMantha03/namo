from __future__ import annotations

from playwright.async_api import Browser, Page, TimeoutError as PlaywrightTimeoutError, async_playwright

from .models import BrowserAction, InteractiveElement, PageObservation


class BrowserController:
    def __init__(self, headless: bool):
        self.headless = headless
        self._playwright = None
        self._browser: Browser | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "BrowserController":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self.page = await self._browser.new_page(viewport={"width": 1280, "height": 900})
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def observe(self) -> PageObservation:
        page = self._require_page()
        await page.evaluate(
            """
            () => {
              const items = Array.from(document.querySelectorAll(
                'a,button,input,textarea,select,[role="button"],[contenteditable="true"]'
              ));
              items.forEach((el, index) => el.setAttribute('data-agent-ref', `e${index + 1}`));
            }
            """
        )
        raw_elements = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll('[data-agent-ref]')).slice(0, 80).map((el) => {
              const tag = el.tagName.toLowerCase();
              const role = el.getAttribute('role') || tag;
              const label = el.getAttribute('aria-label')
                || el.getAttribute('placeholder')
                || el.innerText
                || el.value
                || el.name
                || el.id
                || '';
              return {
                ref: el.getAttribute('data-agent-ref'),
                role,
                text: String(label).replace(/\\s+/g, ' ').trim().slice(0, 160),
                selector_hint: tag
              };
            })
            """
        )
        try:
            text = await page.locator("body").inner_text(timeout=5000)
        except PlaywrightTimeoutError:
            text = ""
        return PageObservation(
            url=page.url,
            title=await page.title(),
            text=" ".join(text.split())[:5000],
            elements=[InteractiveElement(**element) for element in raw_elements if element.get("ref")],
        )

    async def execute(self, action: BrowserAction) -> str:
        page = self._require_page()
        if action.action == "goto":
            if not action.url:
                raise ValueError("goto requires url")
            await page.goto(action.url, wait_until="domcontentloaded")
            return f"Opened {action.url}"
        if action.action == "click":
            locator = self._locator_for_ref(action.ref)
            await locator.click(timeout=7000)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=7000)
            except PlaywrightTimeoutError:
                pass
            return f"Clicked {action.ref}"
        if action.action == "fill":
            if action.value is None:
                raise ValueError("fill requires value")
            locator = self._locator_for_ref(action.ref)
            await locator.fill(action.value, timeout=7000)
            return f"Filled {action.ref}"
        if action.action == "select":
            if action.value is None:
                raise ValueError("select requires value")
            locator = self._locator_for_ref(action.ref)
            await locator.select_option(label=action.value, timeout=7000)
            return f"Selected {action.value} in {action.ref}"
        if action.action == "wait":
            await page.wait_for_timeout(1500)
            return "Waited"
        return action.reason

    def _locator_for_ref(self, ref: str | None):
        if not ref:
            raise ValueError("action requires ref")
        return self._require_page().locator(f'[data-agent-ref="{ref}"]').first

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("Browser is not running")
        return self.page
