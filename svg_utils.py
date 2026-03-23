"""Technical utilities for parsing LLM4SVG-encoded SVG strings."""

import re

COORD_TOKEN = "[COORD]"

SVG_SEMANTIC_TOKENS = [
    "[<|START_OF_SVG|>]", "[<|END_OF_SVG|>]",
    "[<|start_of_g|>]", "[<|end_of_g|>]",
    "[<|svg_path|>]", "[<|svg_circle|>]", "[<|svg_rect|>]", "[<|svg_ellipse|>]",
    "[<|svg_line|>]", "[<|svg_polyline|>]", "[<|svg_polygon|>]",
    "[<|path_d|>]", "[<|moveto|>]", "[<|lineto|>]", "[<|curveto|>]",
    "[<|smooth_curveto|>]", "[<|quadratic_curveto|>]", "[<|smooth_quadratic_curveto|>]",
    "[<|arc|>]", "[<|horizontal_lineto|>]", "[<|vertical_lineto|>]", "[<|close_the_path|>]",
    "[<|fill|>]", "[<|stroke|>]", "[<|stroke-width|>]", "[<|stroke-linecap|>]",
    "[<|stroke-linejoin|>]", "[<|stroke-dasharray|>]", "[<|stroke-miterlimit|>]",
    "[<|fill-opacity|>]", "[<|stroke-opacity|>]", "[<|opacity|>]",
    "[<|fill-rule|>]", "[<|clip-rule|>]",
    "[<|cx|>]", "[<|cy|>]", "[<|r|>]", "[<|rx|>]", "[<|ry|>]",
    "[<|x|>]", "[<|y|>]", "[<|x1|>]", "[<|y1|>]", "[<|x2|>]", "[<|y2|>]",
    "[<|width|>]", "[<|height|>]", "[<|points|>]",
    "[<|transform|>]", "[<|font-size|>]",
    "[<|stop-color|>]", "[<|offset|>]",
    "[<|dx|>]", "[<|dy|>]",
    "[<|defs|>]", "[<|end_of_defs|>]",
    "[<|linearGradient|>]", "[<|end_of_linearGradient|>]",
    "[<|radialGradient|>]", "[<|end_of_radialGradient|>]",
    "[<|stop|>]", "[<|clipPath|>]", "[<|end_of_clipPath|>]",
    "[<|use|>]", "[<|text|>]", "[<|end_of_text|>]",
]

COORDINATE_CONTEXT = {
    "[<|moveto|>]", "[<|lineto|>]", "[<|curveto|>]", "[<|smooth_curveto|>]",
    "[<|quadratic_curveto|>]", "[<|smooth_quadratic_curveto|>]",
    "[<|arc|>]", "[<|horizontal_lineto|>]", "[<|vertical_lineto|>]",
    "[<|cx|>]", "[<|cy|>]", "[<|r|>]", "[<|rx|>]", "[<|ry|>]",
    "[<|x|>]", "[<|y|>]", "[<|x1|>]", "[<|y1|>]", "[<|x2|>]", "[<|y2|>]",
    "[<|width|>]", "[<|height|>]", "[<|points|>]",
    "[<|dx|>]", "[<|dy|>]",
}

_TOKENIZE_RE = re.compile(r'(\[<\|[^|]+\|>\])')
_NUMBER_RE = re.compile(r'^-?\d+\.?\d*$')


def _tokenize_encoded_svg(encoded_svg):
    """Split an LLM4SVG-encoded string into a flat list of semantic tokens,
    numbers, and other literal values (colors, keywords).

    The encoded format concatenates tokens without spaces:
        [<|moveto|>]79.3 120[<|curveto|>]0 2.21 -6.85 4...
    This function splits on semantic token boundaries first, then on whitespace.
    """
    parts = _TOKENIZE_RE.split(encoded_svg)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _TOKENIZE_RE.fullmatch(part):
            tokens.append(part)
        else:
            tokens.extend(part.split())
    return tokens


def parse_encoded_svg(encoded_svg):
    """Parse an LLM4SVG-encoded SVG string into a skeleton with [COORD] placeholders
    and a list of extracted coordinate values.

    Numbers following coordinate-context semantic tokens (path commands, geometric
    attributes) are replaced with [COORD]. Non-coordinate values (colors, opacities,
    stroke widths) are preserved as literals.

    Returns:
        tuple: (skeleton_str, coordinates_list)
    """
    tokens = _tokenize_encoded_svg(encoded_svg)
    skeleton, coords = [], []
    state = "other"

    for tok in tokens:
        if _TOKENIZE_RE.fullmatch(tok):
            skeleton.append(tok)
            if tok in COORDINATE_CONTEXT:
                state = "coord"
            else:
                state = "other"
        elif _NUMBER_RE.match(tok) and state == "coord":
            skeleton.append(COORD_TOKEN)
            coords.append(float(tok))
        else:
            skeleton.append(tok)
            state = "other"

    return " ".join(skeleton), coords


def normalize_coordinates(coords):
    """Normalize coordinates into [-1, 1] space.

    Computes a per-sample offset (center) and scale (half-range) so that the
    full coordinate range maps to [-1, 1].

    Returns:
        tuple: (normalized_coords, offset, scale)
    """
    if not coords:
        return [], 0.0, 1.0
    lo, hi = min(coords), max(coords)
    offset = (hi + lo) / 2.0
    scale = (hi - lo) / 2.0
    if scale == 0.0:
        scale = 1.0
    return [(c - offset) / scale for c in coords], offset, scale


def denormalize_coordinates(coords, offset, scale):
    """Reverse normalization from [-1, 1] back to SVG coordinate space."""
    return [c * scale + offset for c in coords]


def decode_to_svg(encoded_svg, viewbox="0 0 128 128"):
    """Decode an LLM4SVG-encoded string back to real SVG XML.

    Handles paths (with d commands), basic shapes (circle, rect, ellipse, line,
    polyline, polygon), groups, defs, gradients, stops, and common attributes.
    """
    tokens = _tokenize_encoded_svg(encoded_svg)

    PATH_COMMANDS = {
        "[<|moveto|>]": "m", "[<|lineto|>]": "l", "[<|curveto|>]": "c",
        "[<|smooth_curveto|>]": "s", "[<|quadratic_curveto|>]": "q",
        "[<|smooth_quadratic_curveto|>]": "t", "[<|arc|>]": "a",
        "[<|horizontal_lineto|>]": "h", "[<|vertical_lineto|>]": "v",
        "[<|close_the_path|>]": "z",
    }
    SHAPE_ELEMENTS = {
        "[<|svg_path|>]": "path", "[<|svg_circle|>]": "circle",
        "[<|svg_rect|>]": "rect", "[<|svg_ellipse|>]": "ellipse",
        "[<|svg_line|>]": "line", "[<|svg_polyline|>]": "polyline",
        "[<|svg_polygon|>]": "polygon",
    }
    ATTR_MAP = {
        "[<|fill|>]": "fill", "[<|stroke|>]": "stroke",
        "[<|stroke-width|>]": "stroke-width", "[<|stroke-linecap|>]": "stroke-linecap",
        "[<|stroke-linejoin|>]": "stroke-linejoin", "[<|stroke-dasharray|>]": "stroke-dasharray",
        "[<|stroke-miterlimit|>]": "stroke-miterlimit",
        "[<|fill-opacity|>]": "fill-opacity", "[<|stroke-opacity|>]": "stroke-opacity",
        "[<|opacity|>]": "opacity", "[<|fill-rule|>]": "fill-rule",
        "[<|clip-rule|>]": "clip-rule",
        "[<|cx|>]": "cx", "[<|cy|>]": "cy", "[<|r|>]": "r",
        "[<|rx|>]": "rx", "[<|ry|>]": "ry",
        "[<|x|>]": "x", "[<|y|>]": "y",
        "[<|x1|>]": "x1", "[<|y1|>]": "y1", "[<|x2|>]": "x2", "[<|y2|>]": "y2",
        "[<|width|>]": "width", "[<|height|>]": "height",
        "[<|transform|>]": "transform", "[<|font-size|>]": "font-size",
        "[<|stop-color|>]": "stop-color", "[<|offset|>]": "offset",
        "[<|dx|>]": "dx", "[<|dy|>]": "dy",
    }
    # Tokens whose following values should be collected greedily (multiple numbers)
    MULTI_VALUE_ATTRS = {"[<|points|>]", "[<|stroke-dasharray|>]"}

    CONTAINER_OPEN = {
        "[<|start_of_g|>]": "g", "[<|defs|>]": "defs",
        "[<|linearGradient|>]": "linearGradient",
        "[<|radialGradient|>]": "radialGradient",
        "[<|clipPath|>]": "clipPath", "[<|text|>]": "text",
    }
    CONTAINER_CLOSE = {
        "[<|end_of_g|>]", "[<|end_of_defs|>]",
        "[<|end_of_linearGradient|>]", "[<|end_of_radialGradient|>]",
        "[<|end_of_clipPath|>]", "[<|end_of_text|>]",
    }

    xml_parts = []
    elem_tag = None
    elem_attrs = {}
    path_data = []
    in_path_d = False
    multi_values = []
    multi_attr = None
    pending_attr = None

    def _flush_element():
        nonlocal elem_tag, elem_attrs, path_data, in_path_d
        nonlocal multi_values, multi_attr
        if in_path_d and path_data:
            elem_attrs["d"] = " ".join(path_data)
            path_data, in_path_d = [], False
        if multi_attr and multi_values:
            elem_attrs[multi_attr] = " ".join(multi_values)
            multi_attr, multi_values = None, []
        if elem_tag:
            attrs = " ".join(f'{k}="{v}"' for k, v in elem_attrs.items())
            sep = " " if attrs else ""
            xml_parts.append(f"<{elem_tag}{sep}{attrs}/>")
            elem_tag, elem_attrs = None, {}

    for tok in tokens:
        is_semantic = _TOKENIZE_RE.fullmatch(tok)

        # If collecting multi-value attribute, numbers continue it; anything else ends it
        if multi_attr and (is_semantic or not _NUMBER_RE.match(tok)):
            elem_attrs[multi_attr] = " ".join(multi_values)
            multi_attr, multi_values = None, []

        if tok == "[<|START_OF_SVG|>]":
            xml_parts.append(
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}">'
            )
        elif tok == "[<|END_OF_SVG|>]":
            _flush_element()
            xml_parts.append("</svg>")
        elif tok in SHAPE_ELEMENTS:
            _flush_element()
            elem_tag = SHAPE_ELEMENTS[tok]
            elem_attrs = {}
        elif tok == "[<|stop|>]":
            _flush_element()
            elem_tag = "stop"
            elem_attrs = {}
        elif tok == "[<|use|>]":
            _flush_element()
            elem_tag = "use"
            elem_attrs = {}
        elif tok in CONTAINER_OPEN:
            _flush_element()
            xml_parts.append(f"<{CONTAINER_OPEN[tok]}>")
        elif tok in CONTAINER_CLOSE:
            _flush_element()
            # Derive tag name: e.g. "[<|end_of_g|>]" → "g"
            inner = tok[len("[<|end_of_"):-len("|>]")]
            xml_parts.append(f"</{inner}>")
        elif tok == "[<|path_d|>]":
            in_path_d = True
            path_data = []
        elif tok in PATH_COMMANDS:
            if in_path_d:
                path_data.append(PATH_COMMANDS[tok])
        elif tok in MULTI_VALUE_ATTRS:
            if in_path_d and path_data:
                elem_attrs["d"] = " ".join(path_data)
                path_data, in_path_d = [], False
            multi_attr = ATTR_MAP.get(tok, tok.strip("[<|>]"))
            multi_values = []
        elif tok in ATTR_MAP:
            if in_path_d and path_data:
                elem_attrs["d"] = " ".join(path_data)
                path_data, in_path_d = [], False
            pending_attr = ATTR_MAP[tok]
        elif not is_semantic:
            # Plain value (number, color, keyword)
            if multi_attr:
                multi_values.append(tok)
            elif in_path_d:
                path_data.append(tok)
            elif pending_attr:
                elem_attrs[pending_attr] = tok
                pending_attr = None

    _flush_element()
    # Ensure the SVG is always properly closed (skeleton may be truncated)
    if any("<svg" in p for p in xml_parts) and not any("</svg>" in p for p in xml_parts):
        xml_parts.append("</svg>")
    return "\n".join(xml_parts)


def reconstruct_svg(skeleton, coordinates, offset=0.0, scale=1.0):
    """Replace [COORD] placeholders with coordinate values and rejoin tokens
    into the concatenated LLM4SVG format (no spaces between semantic tokens
    and their values).

    If *offset* and *scale* are provided the coordinates are treated as
    normalised values in [-1, 1] and are denormalised first via
    ``denormalize_coordinates``.
    """
    if offset != 0.0 or scale != 1.0:
        coordinates = denormalize_coordinates(coordinates, offset, scale)

    tokens = skeleton.split()
    result, idx = [], 0
    for tok in tokens:
        if tok == COORD_TOKEN:
            result.append(f"{coordinates[idx]:.2f}")
            idx += 1
        else:
            result.append(tok)

    out = []
    for tok in result:
        if out and not _TOKENIZE_RE.fullmatch(tok) and not _TOKENIZE_RE.fullmatch(out[-1]):
            out.append(" ")
        out.append(tok)
    return "".join(out)
