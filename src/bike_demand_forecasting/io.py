import zipfile
from pathlib import Path
import shutil
import pandas as pd


def extract_all_zips(data_dir: Path, extract_dir: Path) -> None:
    """
    extract zip files and store them
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    for path in data_dir.glob("*.zip"):
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extract_dir / path.stem)

        macosx_dir = extract_dir / path.stem / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        print(f"{path}: extracted")



def merge_all_csv(extract_dir: Path, output_csv: Path, encoding: str = "latin-1") -> Path:
    """
    Retrieve all csv files for a year and concat them
    """
    output_csv = output_csv.with_suffix(".csv")  # force .csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {extract_dir}")

    df = pd.concat([pd.read_csv(f, encoding=encoding) for f in csv_files], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Merged file created: {output_csv}")

    return output_csv
