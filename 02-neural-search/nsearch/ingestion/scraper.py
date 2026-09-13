import asyncio
import itertools
from pathlib import Path
from nsearch.ingestion.constants import BASE_URL, CONFIG_FILE, DEST_DIR
from nsearch.ingestion.source_config import EntscheidSuche, generate_valid_areas
import aiohttp
import time
from bs4 import BeautifulSoup

LIMIT_SEMAPHORE = 50

def build_folder(area, topic) -> str:
    return f"{area}_{topic}"

def extract_folders(data_model: EntscheidSuche) -> set[str]:
    return {
        build_folder(area, topic)
        for area, topics in data_model.areas.items()
        for topic in topics
    }

async def ingest_raw_files(folders: set[str], dest_path: str):
    sem = asyncio.Semaphore(LIMIT_SEMAPHORE)
    timeout = aiohttp.ClientTimeout(connect=10, sock_read=30, total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def list_folder(folder: str) -> tuple[str, list[str], str | None]:
            """Phase 1: list one folder's files. Returns (folder, file_names, error)."""
            url = BASE_URL + folder + "/"
            folder_dir = Path(dest_path) / folder
            try:
                async with sem, session.get(url) as resp:
                    resp.raise_for_status()
                    html = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                return folder, [], str(e)

            soup = BeautifulSoup(html, "lxml")
            file_names = [
                href for link in soup.find_all("a", href=True)
                if isinstance(href := link["href"], str) and not href.endswith("/")
            ]
            folder_dir.mkdir(parents=True, exist_ok=True)
            return folder, file_names, None

        async with asyncio.TaskGroup() as tg:
            listing_tasks = [tg.create_task(list_folder(f)) for f in folders]
        listings = [t.result() for t in listing_tasks]
        failed_folders = [(f, err) for f, _, err in listings if err]

        start_time = time.perf_counter()
        progress = {"count": 0}

        async def download_file(folder: str, name: str) -> tuple[str, str, str] | None:
            """Phase 2: download one file. Returns None on success, else (folder, name, error)."""
            #NOTE: Styled as a closure: function defined inside another function
            file_path = Path(dest_path) / folder / name
            if file_path.exists():
                return None
            try:
                async with sem, session.get(BASE_URL + folder + "/" + name) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
                file_path.write_bytes(content)
                progress["count"] += 1  # safe without a lock: single-threaded event loop
                if progress["count"] % 1000 == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = progress["count"] / elapsed
                    print(f"{progress['count']} files in {elapsed:.1f}s ({rate:.1f} files/sec)")
                return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                return folder, name, str(e)

        async with asyncio.TaskGroup() as tg:
            # Scrape the files per folder in round-robin fashion
            per_folder = [
                [(folder, name) for name in file_names] # file_names[:N] to limit items
                for folder, file_names, err in listings if not err
            ]
            round_robin = [
                pair for group in itertools.zip_longest(*per_folder)
                for pair in group if pair is not None
            ]
            download_tasks = [tg.create_task(download_file(folder, name)) for (folder, name) in round_robin]
            print(f"There are {len(download_tasks)} created, one per file")
        failed_files = [r for r in (t.result() for t in download_tasks) if r is not None]

        return failed_folders, failed_files


async def main():
    valid_areas = await generate_valid_areas() # sequential
    ents_suche = EntscheidSuche.from_yaml(CONFIG_FILE, valid_areas=valid_areas)
    print(ents_suche)

    required_folders = extract_folders(ents_suche)
    print(required_folders, DEST_DIR)
    # Parallelize
    failed_folders, failed_files = await ingest_raw_files(folders=required_folders, dest_path=DEST_DIR)
    for f in failed_folders:
        print(f"The following folder has failed completely: {f}")
        #TODO: add retry
    for f in failed_files:
        print(f"The following files have failed inside their folder: {f}")
        #TODO: add retry


if __name__ == "__main__":
    asyncio.run(main(), debug=True)


"""
Other way of writing it, dirtier:

if __name__ == "__main__":
    valid_areas = asyncio.run(generate_valid_areas())
    ents_suche = EntscheidSuche.from_yaml(CONFIG_FILE, valid_areas=valid_areas)
    print(ents_suche)

    required_folders = extract_folders(ents_suche)
    asyncio.run(ingest_raw_files(folders=required_folders, dest_path=DEST_DIR))

* Each asyncio.run() spins up a fresh event loop and tears it down when that call returns. 
* Two calls means two loops, created and destroyed back to back: repeated setup/teardown cost.
* You lose the ability to share async resources across the calls.
"""