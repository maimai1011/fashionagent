"""Conservative garment-type normalization for wardrobe retrieval."""

import re
import unicodedata

_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("t-shirt", ("t shirt", "tshirt", "tee", "tees", "ao thun", "ao phong")),
    ("shirt", ("shirt", "shirts", "button down", "button up", "oxford", "ao so mi")),
    ("blouse", ("blouse", "blouses", "ao kieu")),
    ("hoodie", ("hoodie", "hoodies", "sweatshirt", "sweatshirts", "ao ni")),
    ("sweater", ("sweater", "sweaters", "jumper", "jumpers", "cardigan", "cardigans", "ao len")),
    ("jacket", ("jacket", "jackets", "blazer", "blazers", "ao khoac")),
    ("coat", ("coat", "coats", "parka", "parkas", "trench", "ao mang to")),
    ("dress", ("dress", "dresses", "gown", "gowns", "dam", "vay lien", "vay")),
    ("skirt", ("skirt", "skirts", "chan vay")),
    ("jeans", ("jeans", "denim pants", "quan jean", "quan bo")),
    ("pants", ("pants", "trousers", "trouser", "slacks", "leggings", "quan dai")),
    ("shorts", ("shorts", "short", "quan short", "quan dui")),
    ("shoes", (
        "shoes", "shoe", "sneakers", "sneaker", "boots", "boot", "heels",
        "heel", "sandals", "sandal", "loafers", "loafer", "flats", "footwear",
        "giay", "giay the thao",
    )),
    ("watch", ("watch", "watches", "timepiece", "dong ho")),
    ("bag", (
        "bag", "bags", "handbag", "handbags", "backpack", "backpacks",
        "purse", "purses", "tui xach", "tui",
    )),
    ("jewelry", (
        "jewelry", "jewellery", "necklace", "bracelet", "earrings", "ring",
        "trang suc", "vong tay", "day chuyen",
    )),
    ("activewear", ("activewear", "sportswear")),
    ("outerwear", ("outerwear",)),
    ("bottom", ("bottom", "bottoms", "quan")),
    ("top", ("top", "tops", "ao")),
    ("accessory", ("accessory", "accessories", "phu kien")),
)

_COLORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("navy", ("navy", "xanh navy", "xanh dam")),
    ("blue", ("blue", "xanh duong")),
    ("green", ("green", "xanh la")),
    ("red", ("red", "do")),
    ("black", ("black", "den")),
    ("white", ("white", "trang")),
    ("yellow", ("yellow", "vang")),
    ("pink", ("pink", "hong")),
    ("purple", ("purple", "tim")),
    ("brown", ("brown", "nau")),
    ("gray", ("gray", "grey", "xam")),
    ("orange", ("orange", "cam")),
    ("beige", ("beige", "be")),
)

_FAMILIES = {
    "top": {"t-shirt", "shirt", "blouse", "hoodie", "sweater", "top"},
    "bottom": {"skirt", "jeans", "pants", "shorts", "bottom"},
    "outerwear": {"jacket", "coat", "outerwear"},
    "accessory": {"watch", "bag", "jewelry", "accessory"},
}


def _fold(text: str) -> str:
    ascii_text = "".join(
        char for char in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def _match(text: str, aliases: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    words = _fold(text)
    found = [
        (len(alias.split()), len(alias), kind)
        for kind, names in aliases for alias in names
        if re.search(rf"\b{re.escape(alias)}\b", words)
    ]
    return max(found)[2] if found else None


def normalize_category(category: str, item_name: str = "") -> str | None:
    """Use a specific name when available, then fall back to category metadata."""
    return _match(item_name, _ALIASES) or _match(category, _ALIASES)


def normalize_color(color: str) -> str:
    """Normalize common English and Vietnamese color labels for exact filters."""
    return _match(color, _COLORS) or _fold(color)


def wardrobe_display_name(category: str, color: str, item_name: str = "") -> str:
    """Describe an upload using verified metadata without exposing its filename."""
    kind = normalize_category(category, item_name) or category.strip() or "wardrobe item"
    shade = normalize_color(color) if color.strip() else ""
    if shade in {"unknown", "other", "none"}:
        shade = ""
    return f"{shade} {kind}".strip().capitalize()


def category_compatible(
    requested_category: str,
    requested_name: str,
    available_category: str,
    available_name: str,
) -> bool:
    """Reject unknown and incompatible types; generic requests allow their family."""
    requested = normalize_category(requested_category, requested_name)
    available = normalize_category(available_category, available_name)
    if requested is None or available is None:
        return False
    if requested == available:
        return True
    return requested in _FAMILIES and available in _FAMILIES[requested]
