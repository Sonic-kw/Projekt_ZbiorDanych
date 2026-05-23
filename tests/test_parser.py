from src.scraper.parser import OtoMotoParser


class _FakeElement:
    def __init__(self, text: str = "", attrs: dict | None = None):
        self._text = text
        self._attrs = attrs or {}

    def inner_text(self) -> str:
        return self._text

    def get_attribute(self, name: str) -> str | None:
        return self._attrs.get(name)


class _FakeItem:
    def __init__(self, selectors: dict[str, _FakeElement]):
        self._selectors = selectors

    def query_selector(self, selector: str):
        return self._selectors.get(selector)


class _FakePage:
    def __init__(self, items: list[_FakeItem]):
        self._items = items

    def query_selector_all(self, selector: str):
        if selector == "article.ooa-zet1mn":
            return self._items
        return []


def test_parser_extracts_listing_url_and_id():
    parser = OtoMotoParser()
    item = _FakeItem(
        {
            "h2.e123dwbo0 a": _FakeElement(
                text="Honda CB500",
                attrs={"href": "/motocykle-i-quady/oferta/honda-cb500-ID6Hh123.html"},
            ),
            "h3.eg88ra81": _FakeElement(text="20 000 PLN"),
            'dd[data-parameter="mileage"]': _FakeElement(text="10 000 km"),
            'dd[data-parameter="year"]': _FakeElement(text="2020"),
            'dd[data-parameter="engine_capacity"]': _FakeElement(text="500 cm3"),
            "p.e1kj25my0": _FakeElement(text="500 cm3 • 47 KM • Benzyna"),
        }
    )

    listings = parser.parse_listings(_FakePage([item]))

    assert len(listings) == 1
    assert listings[0]["listing_id"] == "ID6Hh123"
    assert listings[0]["listing_url"] == "https://www.otomoto.pl/motocykle-i-quady/oferta/honda-cb500-ID6Hh123.html"
