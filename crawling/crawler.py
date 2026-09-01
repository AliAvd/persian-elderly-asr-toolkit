import contextlib
import csv
import os
import re
import wave

import requests
from bs4 import BeautifulSoup

BASE_URL = os.getenv("GANJOOR_BASE_URL", "https://ganjoor.net/hafez/ghazal/sh{}")
SAVE_DIR = os.getenv("GANJOOR_OUTPUT_DIR", "ganjoor_audio")
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(SAVE_DIR, "audio_xml_map.csv")

# Initialize CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sh_number", "page_url", "audio_file", "xml_file"])

XML_PUSH_PATTERN = re.compile(r"audioxmlfiles\.push\((?:'|\")(.+?)(?:'|\")\)")
NARRATOR_PUSH_PATTERN = re.compile(r"narrators\.push\((?:'|\")(.+?)(?:'|\")\)")


def extract_first(taglist):
    return taglist[0] if taglist else None


def safe_filename(url):
    if not url:
        return None
    return url.split("/")[-1]


def duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration * 1000)


def extract_b_blocks(url):
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find("div", class_="poem")  # where b/b2 occur
    if not container:
        print("No poem container found.")
        return

    blocks = []

    # for div in container.find_all("div", recursive=True):
    #     cls = div.get("class", [])
    #     if "b" in cls or "b2" in cls:
    #         text = div.get_text(strip=True)
    #         if text:
    #             blocks.append(text)

    # return blocks
    for div in container.find_all("div", recursive=True):
        cls = div.get("class", [])
        if "b" in cls or "b2" in cls:
            # find all <p> inside this div
            ps = div.find_all("p", recursive=False)

            # fallback: sometimes p is nested
            if not ps:
                ps = div.find_all("p")

            # texts = [p.get_text(strip=True) for p in ps if p.get_text(strip=True)]
            for p in ps:
                text = p.get_text(strip=True)
                if text:
                    blocks.append(text)

            # if texts:
            #     blocks.append(" ".join(texts))

    return blocks


import xml.etree.ElementTree as ET


def parse_xml_even_ms(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ms_values = []

    for sync in root.findall(".//SyncInfo"):
        order = int(sync.find("VerseOrder").text.strip())
        if order >= 0:
            ms = int(sync.find("AudioMiliseconds").text.strip())
            ms_values.append(ms)

    # # take 2nd, 4th, 6th, ...
    # even_ms = ms_values[1::2]

    # return even_ms
    return ms_values


import os

from pydub import AudioSegment


def chunk_audio(audio_path, ms_list, out_dir, prefix):
    audio = AudioSegment.from_file(audio_path)
    os.makedirs(out_dir, exist_ok=True)

    chunks = []
    start = ms_list[0]

    for i, end in enumerate(ms_list):
        if i == 0:
            continue
        chunk = audio[start:end]
        chunk_path = os.path.join(out_dir, f"{prefix}_chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)
        start = end

    return chunks


import csv


def write_alignment_csv(csv_path, texts, audio_chunks, narrator):
    # print(texts)
    # print(audio_chunks)
    # audio_chunks.pop(0)
    texts.pop(-1)
    assert len(texts) == len(audio_chunks), "Mismatch between b-blocks and audio chunks"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "text", "narrator"])

        for audio, text in zip(audio_chunks, texts):
            writer.writerow([audio, text, narrator])


def process_poem_alignment(url, audio_path, xml_path, narrator, poem_id):
    # 1) text
    b_blocks = extract_b_blocks(url)

    # 2) timing
    even_ms = parse_xml_even_ms(xml_path)

    # 3) audio chunks
    chunk_dir = f"chunks/{poem_id + 110}"
    chunks = chunk_audio(audio_path, even_ms, chunk_dir, poem_id)

    # 4) CSV
    csv_path = f"chunks/{poem_id + 110}/alignment.csv"
    write_alignment_csv(csv_path, b_blocks, chunks, narrator)

    print(f"✅ Alignment created: {csv_path}")


def crawl_page(sh_number):
    url = BASE_URL.format(sh_number)
    print(f"\n### Crawling {url}")

    r = requests.get(url, timeout=100)
    if r.status_code != 200:
        print("  -> page not found.")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # ----- Find audio from HTML -----
    first_audio_src = None
    first_narrator = None
    audio_players = soup.find_all("div", class_="audio-player")
    if audio_players:
        audio_tag = audio_players[0].find("audio")
        if audio_tag:
            source_tag = audio_tag.find("source")
            if source_tag and source_tag.has_attr("src"):
                first_audio_src = source_tag["src"]

    # ----- Find XML in inline/external JS -----
    first_xml_url = None
    script_tags = soup.find_all("script")

    for sc in script_tags:
        if sc.has_attr("src"):
            js_url = sc["src"]
            if js_url.startswith("//"):
                js_url = "https:" + js_url
            elif js_url.startswith("/"):
                js_url = "https://ganjoor.net" + js_url

            try:
                js_text = requests.get(js_url, timeout=10).text
            except requests.RequestException:
                continue

            xml_match = XML_PUSH_PATTERN.search(js_text)
            narrator_match = NARRATOR_PUSH_PATTERN.search(js_text)
            if xml_match and narrator_match:
                first_xml_url = xml_match.group(1)
                first_narrator = narrator_match.group(1).strip()
                break

        else:
            if sc.string:
                xml_match = XML_PUSH_PATTERN.search(sc.string)
                narrator_match = NARRATOR_PUSH_PATTERN.search(sc.string)
                if xml_match and narrator_match:
                    first_xml_url = xml_match.group(1)
                    first_narrator = narrator_match.group(1).strip()
                    break

            inline_text = sc.get_text()
            xml_match = XML_PUSH_PATTERN.search(inline_text)
            narrator_match = NARRATOR_PUSH_PATTERN.search(inline_text)
            if xml_match and narrator_match:
                first_xml_url = xml_match.group(1)
                first_narrator = narrator_match.group(1).strip()
                break

    return {
        "url": url,
        "audio_src_html": first_audio_src,
        "audio_xml_url": first_xml_url,
        "narrator": first_narrator,
    }


def download(url):
    """Download and return saved filename (or None)."""
    if not url:
        return None

    filename = safe_filename(url)
    path = os.path.join(SAVE_DIR, filename)

    try:
        print(f"  -> downloading {url}")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"     saved: {path}")
        return filename
    except Exception as e:
        print("     download failed:", e)
        return None


# ======================
#        MAIN
# ======================

for sh in range(2, 400):  # increase range if needed
    data = crawl_page(sh)
    if not data:
        break
    audio_file = download(data["audio_src_html"])
    xml_file = download(data["audio_xml_url"])

    if audio_file is None or xml_file is None:
        break
    else:
        if data["narrator"] is None:
            narrator = "رندوم"
        else:
            narrator = data["narrator"]

    audio_file = os.path.join(SAVE_DIR, audio_file)
    xml_file = os.path.join(SAVE_DIR, xml_file)

    # # write CSV mapping
    # with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         sh,
    #         data["url"],
    #         audio_file,
    #         xml_file
    #     ])
    # print(audio_file, xml_file)
    url = BASE_URL.format(sh)
    process_poem_alignment(url, audio_file, xml_file, narrator, sh)

    print(f"  -> CSV updated for sh{sh}")
