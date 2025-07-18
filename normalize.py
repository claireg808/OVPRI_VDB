## Clean .txt files
    # Removing extra whitespace
    # Removing special characters
    # Removing page numbers
    # Removing boilerplate text
    # Converting to lowercase
    # Standardize dates: MM/DD/YYYY

import os
import re
from pathlib import Path
from cleantext import clean
from datetime import datetime


# read in regex removal lines
def load_regex_patterns():
    with open('regex_remove.txt', 'r', encoding='utf-8') as file:
        return file.read().splitlines()


# apply regex to remove custom lines
def remove_custom(text, regex_lines):
    for rgx in regex_lines:
        try:
            text = re.sub(rgx, "", text)
        except re.error as e:
            print(f"[ERROR] Invalid regex pattern '{rgx}': {e}")
    text = re.sub(r'(\n\s*){2,}', '\n', text, flags=re.MULTILINE)
    return text


# precompile regexes for speed
_RE_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")  # YYYY-MM-DD
_RE_DMY = re.compile(r"\b(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})\b")  # DD-MM-YYYY
_RE_MONTH = re.compile(r"\b(?P<mon>[A-Za-z]+)\s(?P<d>\d{1,2}),\s(?P<y>\d{4})\b")  # Month DD, YYYY


# convert to MM/DD/YYYY format
def _to_mmddyyyy(year: int, month: int, day: int) -> str:
    try:
        dt = datetime(year, month, day)
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        # invalid calendar date- force format
        return f"{month:02d}/{day:02d}/{year:04d}"


# convert named month
def _month_name_to_num(name: str) -> int | None:
    try:
        return datetime.strptime(name.strip()[:3], "%b").month  # first 3 chars enough
    except ValueError:
        # try full month name
        try:
            return datetime.strptime(name.strip(), "%B").month
        except ValueError:
            return None


# scan text for dates to convert
def date_conversion(text: str) -> str:
    # YYYY-MM-DD
    def repl_iso(m: re.Match) -> str:
        y = int(m.group("y"))
        mo = int(m.group("m"))
        d = int(m.group("d"))
        return _to_mmddyyyy(y, mo, d)

    # DD-MM-YYYY
    def repl_dmy(m: re.Match) -> str:
        d = int(m.group("d"))
        mo = int(m.group("m"))
        y = int(m.group("y"))
        return _to_mmddyyyy(y, mo, d)

    # Month DD, YYYY  (full or 3 digit month names)
    def repl_month(m: re.Match) -> str:
        mon_name = m.group("mon")
        d = int(m.group("d"))
        y = int(m.group("y"))
        mo = _month_name_to_num(mon_name)
        if mo is None:
            return m.group(0)  # fallback unchanged
        return _to_mmddyyyy(y, mo, d)


    # apply conversions
    text = _RE_ISO.sub(repl_iso, text)
    text = _RE_DMY.sub(repl_dmy, text)
    text = _RE_MONTH.sub(repl_month, text)

    return text


# normalize the text
def clean_text(text: str, regex_lines: list) -> str:    
    # use cleantext library for initial cleaning
    cleaned_text = clean(text)
    
    # apply custom regex removal
    cleaned_text = remove_custom(cleaned_text, regex_lines)
    
    # apply date conversion
    cleaned_text = date_conversion(cleaned_text)
    
    return cleaned_text


# read file to be normalized
def process_file(input_path: str, output_path: str, regex_lines: list) -> None:
    try:
        # read the file
        with open(input_path, 'r', encoding='utf-8') as fin:
            raw_text = fin.read()
        
        # return the normalized text
        cleaned_text = clean_text(raw_text, regex_lines)
        
        # write normalized text to output
        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write(cleaned_text)
            
        print(f'Successfully processed: {output_path}')
        
    except Exception as e:
        print(f'Error processing {input_path}: {e}')


if __name__ == "__main__":
    # load regex patterns
    regex_lines = load_regex_patterns()
    
    # I/O folders
    input_folder = 'HRPP_text'
    output_folder = 'HRPP_normalized'
    os.makedirs(output_folder, exist_ok=True)
    
    # find all .txt files
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # process each file
    for filename in txt_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_file(input_path, output_path, regex_lines)
    
    print("\n\nComplete")