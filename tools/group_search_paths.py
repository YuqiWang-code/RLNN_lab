"""Group each stage of the search diagram for editing in PowerPoint."""

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as ET


P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def name(shape):
    return shape.find(P + "nvSpPr/" + P + "cNvPr").get("name")


def group_stages(tree):
    next_id = 1 + max(int(node.get("id")) for node in tree.iter(P + "cNvPr"))
    shapes = tree.findall(P + "sp")
    bodies = [shape for shape in shapes if name(shape).endswith(" body")]
    for body in bodies:
        prefix = name(body).removesuffix(" body")
        members = [shape for shape in shapes if name(shape).startswith(prefix + " ")
                   and "publication" not in name(shape).lower()]
        ids = {shape.find(P + "nvSpPr/" + P + "cNvPr").get("id") for shape in members}
        connections = []
        for connection in tree.findall(P + "cxnSp"):
            endpoints = list(connection.iter(A + "stCxn")) + list(connection.iter(A + "endCxn"))
            if len(endpoints) == 2 and all(endpoint.get("id") in ids for endpoint in endpoints):
                connections.append(connection)

        group = ET.Element(P + "grpSp")
        props = ET.SubElement(group, P + "nvGrpSpPr")
        ET.SubElement(props, P + "cNvPr", id=str(next_id), name=prefix)
        next_id += 1
        ET.SubElement(props, P + "cNvGrpSpPr")
        ET.SubElement(props, P + "nvPr")
        transform = ET.SubElement(ET.SubElement(group, P + "grpSpPr"), A + "xfrm")
        bounds = body.find(P + "spPr/" + A + "xfrm")
        for child, original in (("off", "off"), ("ext", "ext"), ("chOff", "off"), ("chExt", "ext")):
            ET.SubElement(transform, A + child, **bounds.find(A + original).attrib)

        tree.insert(tree.index(body), group)
        cards = [shape for shape in members if name(shape).startswith(prefix + " state ")]
        group.extend([body, *cards, *connections])
        group.extend(shape for shape in members if shape is not body and shape not in cards)

    frames = [shape for shape in shapes if name(shape).endswith(" frame")]
    for frame in reversed(frames):
        tree.remove(frame)
        tree.insert(2, frame)


def arrange(directory):
    slide = "ppt/slides/slide1.xml"
    with ZipFile(directory / "candidate.pptx") as source, ZipFile(
        directory / "search_paths.pptx", "w", ZIP_DEFLATED
    ) as output:
        document = ET.fromstring(source.read(slide))
        tree = document.find(".//" + P + "spTree")
        for shape in tree.findall(P + "sp"):
            preset = shape.find(P + "spPr/" + A + "prstGeom")
            if preset is not None and preset.get("prst") == "roundRect":
                guide = preset.find(A + "avLst/" + A + "gd")
                guide.set("fmla", "val 2600" if name(shape).endswith(" frame") else "val 7500")
        group_stages(tree)
        for part in source.infolist():
            data = (ET.tostring(document, encoding="UTF-8", xml_declaration=True, standalone=True)
                    if part.filename == slide else source.read(part.filename))
            output.writestr(part, data)


if __name__ == "__main__":
    arrange(Path(sys.argv[1]))
