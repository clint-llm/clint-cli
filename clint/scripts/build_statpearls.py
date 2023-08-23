import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import *
from xml.etree.ElementTree import Element

LICENSE_CC_BY_NC_ND_4 = """\
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. \
To view a copy of this license, \
visit http://creativecommons.org/licenses/by-nc-nd/4.0/ or send a letter to Creative Commons, \
PO Box 1866, Mountain View, CA 94042, USA.\
"""


class Section(NamedTuple):
    id: str
    title: str
    contents: str


class Article(NamedTuple):
    id: str
    title: str
    sections: List[Section]
    copyright: str
    license: str


def article_id_to_url(article_id: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/books/n/statpearls/{article_id}/"


def section_id_to_url(article_id: str, section_id: str) -> str:
    return article_id_to_url(article_id) + f"#{section_id}"


def convert_xml_to_markdown(xml_filepath: Path) -> Optional[Article]:
    tree = ET.parse(str(xml_filepath))
    root = tree.getroot()

    if root.tag != "book-part-wrapper":
        return None
    article_id = root.attrib.get("id")
    if article_id is None:
        return None

    copyright_elem = root.find("./book-meta/permissions/copyright-statement")
    if copyright_elem is None:
        return None
    copyright = ET.tostring(copyright_elem, encoding="unicode", method="text").strip()

    license_elem = root.find("./book-meta/permissions/license")
    if license_elem is None:
        return None
    license_url = license_elem.attrib.get("{http://www.w3.org/1999/xlink}href")
    if license_url != "https://creativecommons.org/licenses/by-nc-nd/4.0/":
        return None
    license = LICENSE_CC_BY_NC_ND_4

    book_part_elem = root.find(".//book-part[@book-part-type='chapter']")
    if book_part_elem is None:
        return None

    title_group_elem: Optional[Element] = book_part_elem.find(".//title-group")
    if title_group_elem is None:
        return None
    title_elem: Optional[Element] = title_group_elem.find("title")
    if title_elem is None or title_elem.text is None:
        return None
    body_elem: Optional[Element] = book_part_elem.find(".//body")
    if body_elem is None:
        return None

    for element in body_elem.iter():
        if element.tag == "list-item":
            element.text = "\n- {}".format(element.text.strip() if element.text else "")
        if element.tag == "list":
            element.tail = "\n{}".format(element.tail if element.tail else "")
        if element.tag == "bold":
            element.text = "**{}".format(element.text.strip() if element.text else "")
            element.tail = "**{}".format(element.tail.strip() if element.tail else "")
        if element.tag == "xref":
            element.clear()
        if element.tag == "ext-link":
            element.clear()

    sections: List[Section] = []
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_contents: List[str] = []

    # Iterate through each child element in the body
    elements: List[Element] = [body_elem] if body_elem else []
    while elements:
        element = elements.pop()
        if element.tag == "sec":
            if section_id and section_title and section_contents:
                sections.append(
                    Section(section_id, section_title, "".join(section_contents))
                )
            section_id = element.attrib.get("id")
            section_title = None
            section_contents = []
        if element.tag == "title":
            section_title = ET.tostring(
                element, encoding="unicode", method="text"
            ).strip()
            continue
        if element.tag == "p" or element.tag == "list-item":
            contents = ET.tostring(element, encoding="unicode", method="text").strip()
            if element.tag == "list-item" and contents == "-":
                continue
            section_contents.append(f"{contents}\n\n")
            continue
        elements += list(reversed(element))

    if section_id and section_title and section_contents:
        sections.append(Section(section_id, section_title, "".join(section_contents)))

    return Article(article_id, title_elem.text, sections, copyright, license)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    input_folder: Path = args.input_folder
    output_folder: Path = args.output_folder

    output_folder.mkdir(parents=True, exist_ok=True)
    book_path = output_folder / "StatPearls"
    book_parts_path = book_path.with_suffix(".parts")

    path_book_parts: List[str] = []

    for filename in sorted(input_folder.iterdir()):
        if not filename.is_file():
            continue
        if filename.suffix != ".nxml":
            continue

        article = convert_xml_to_markdown(filename)
        if article is None:
            continue

        article_path = book_parts_path / article.title.replace("/", "&")
        article_meta_path = article_path.with_suffix(".meta.json")
        article_parts_path = article_path.with_suffix(".parts")
        article_parts_path.mkdir(parents=True, exist_ok=True)

        path_article_parts: List[str] = []

        for section in article.sections:
            part_path = article_parts_path / section.title.replace("/", "&")
            part_meta_path = part_path.with_suffix(".meta.json")
            part_content_path = part_path.with_suffix(".md")
            with part_content_path.open("w") as fio:
                fio.write(section.contents.strip())
            with part_meta_path.open("w") as fio:
                json.dump(
                    {
                        "title": section.title,
                        "url": section_id_to_url(article.id, section.id),
                        "content": str(part_content_path.relative_to(part_path.parent)),
                    },
                    fio,
                    indent="\t",
                )
            path_article_parts.append(str(part_path.relative_to(article_path.parent)))

        path_book_parts.append(str(article_path.relative_to(book_path.parent)))

        with article_meta_path.open("w") as fio:
            json.dump(
                {
                    "title": article.title,
                    "url": article_id_to_url(article.id),
                    "copyright": article.copyright,
                    "license": article.license,
                    "parts": path_article_parts,
                },
                fio,
                indent="\t",
            )

    book_meta_path = book_path.with_suffix(".meta.json")
    with book_meta_path.open("w") as fio:
        json.dump(
            {
                "title": "StatPearls",
                "url": "https://www.ncbi.nlm.nih.gov/books/n/statpearls/",
                "parts": path_book_parts,
            },
            fio,
            indent="\t",
        )


if __name__ == "__main__":
    main()
