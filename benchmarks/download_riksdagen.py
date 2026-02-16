import os
import re
import unicodedata
from pathlib import Path

import requests
from tqdm import tqdm


def get_riksdagen_data(dok_id: str) -> dict:
    """
    Get riksdagen data for a specific debate

    1. Debate metadata
    2. Speeches (text and metadata)
    """
    response = requests.get(
        f"https://data.riksdagen.se/dokumentstatus/{dok_id}.json?utformat=json&utdata=debatt,media"
    )
    data = response.json()

    return data


def get_riksdagen_audio(audio_url: str, output_dir: str | Path):
    """
    Download riksdagen audio file using streaming requests with tqdm progress bar.
    """

    filename = audio_url.split("/")[-1]
    output_path = Path(output_dir) / filename
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    speeches_media = requests.get(audio_url, stream=True, verify=False)
    total_size = int(speeches_media.headers.get("content-length", 0))
    block_size = 1024

    if speeches_media.status_code == 200:
        with tqdm(total=total_size, unit="B", unit_scale=True, leave=False) as pbar:
            with open(output_path, "wb") as f:
                for data in speeches_media.iter_content(block_size):
                    pbar.update(len(data))
                    f.write(data)
    else:
        raise Exception(f"Failed to download audio: {speeches_media.status_code}")

    return output_path


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub("</p>", "", text)
    text = re.sub(r"\([^\)]*\)", "", text)
    text = re.sub(r"<p>", "\n", text)
    text = text.strip()
    text = text.split("\n")
    text = "\n\n ".join(text)
    return text


def preprocess_debate_metadata(data: dict) -> dict:
    doc_id = data["dokumentstatus"]["debatt"]["anforande"][0]["dok_id"]
    audio_url = data["dokumentstatus"]["webbmedia"]["media"]["audiofileurl"]
    debate_url = data["dokumentstatus"]["webbmedia"]["media"]["debateurl"]
    debate_type = data["dokumentstatus"]["webbmedia"]["media"]["debatt_typ"]

    return {
        "doc_id": doc_id,
        "audio_url": audio_url,
        "debate_url": debate_url,
        "debate_type": debate_type,
    }


def preprocess_speeches(data: dict) -> list[dict]:
    speeches = data["dokumentstatus"]["debatt"]["anforande"]

    all_speeches = []
    for speech in speeches:
        text = clean_text(speech["anf_text"])
        speech_id = f"{speech['dok_id']}-{speech['anf_nummer']}"
        metadata = {
            "doc_id": speech["dok_id"],
            "speech_nr": speech["anf_nummer"],
            "speaker": speech["talare_kort"],
            "party": speech["parti"],
            "date": speech["anf_datum"],
            "datetime": speech["datumtid"],
            "debate_type": speech["debatt_typ"],
            "speech_type": speech["anf_typ"],
            "intressent_id": speech["intressent_id"],
            "intressent_id2": speech["intressent_id2"],
        }

        all_speeches.append(
            {
                "speech_id": speech_id,
                "metadata": metadata,
                "text": text,
                "start": speech["startpos"],
                "end": speech["startpos"] + speech["anf_sekunder"],
            }
        )

    return all_speeches


if __name__ == "__main__":
    dok_ids = ["hdc120251112pd", "hd1092", "hd1081", "hd1025", "hd01ku2"]

    for doc_id in dok_ids:
        data = get_riksdagen_data(doc_id)
        audio_path = get_riksdagen_audio(
            audio_url=data["dokumentstatus"]["webbmedia"]["media"]["audiofileurl"],
            output_dir="data/benchmarks",
        )
        metadata = preprocess_debate_metadata(data)
        speeches = preprocess_speeches(data)
