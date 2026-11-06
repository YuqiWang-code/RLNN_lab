#!/usr/bin/env python3
"""Export model sources and workbook cells to a versioned UTF-8 text snapshot."""
import argparse
import csv
import io
from datetime import datetime
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {'.py', '.pyx', '.pxd', '.c', '.cpp', '.h', '.hpp', '.sh', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.md'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook_text(filename):
    with ZipFile(filename) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(item.itertext()) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {item.attrib['Id']: item.attrib['Target'] for item in rels}
        for sheet in workbook.find('s:sheets', NS):
            rel_id = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rel_id]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            yield '\n### Worksheet: ' + sheet.attrib['name'] + '\n'
            for row in ET.fromstring(archive.read(target)).findall('s:sheetData/s:row', NS):
                fields = []
                for cell in row:
                    reference = cell.attrib['r']
                    letters = ''.join(c for c in reference if c.isalpha())
                    column = 0
                    for letter in letters:
                        column = column * 26 + ord(letter) - ord('A') + 1
                    while len(fields) < column:
                        fields.append('')
                    value = cell.findtext('s:v', default='', namespaces=NS)
                    if cell.attrib.get('t') == 's':
                        value = strings[int(value)]
                    elif cell.attrib.get('t') == 'inlineStr':
                        value = ''.join(cell.find('s:is', NS).itertext())
                    formula = cell.findtext('s:f', namespaces=NS)
                    fields[column - 1] = f'={formula} [value={value}]' if formula is not None else value
                output = io.StringIO()
                csv.writer(output, delimiter='\t', lineterminator='\n').writerow(fields)
                yield output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--workbook', type=Path, default=ROOT / 'docs/实验结果.xlsx')
    args = parser.parse_args()
    if not args.version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.version):
        parser.error('version must contain only letters, numbers, underscores and hyphens')
    workbook = args.workbook.resolve(strict=True)
    workbooks = [workbook]
    smoke = ROOT / 'docs/冒烟结果.xlsx'
    if smoke.exists() and smoke.resolve() != workbook:
        workbooks.append(smoke.resolve())
    now = datetime.now().astimezone()
    target = ROOT / 'docs/temporary' / f'{args.version}_{now:%Y%m%d_%H%M%S}.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    with target.open('x', encoding='utf-8', newline='\n') as output:
        output.write(f'RLNN Lab | {args.version}\nTime: {now.isoformat()}\n')
        if commit.returncode == 0:
            dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'models', 'train_scripts', str(workbook)], cwd=ROOT, text=True)
            output.write(f'Git base: {commit.stdout.strip()}\nWorking tree: {"modified" if dirty.strip() else "clean"}\n')
        for folder in ('models', 'train_scripts'):
            for source in sorted((ROOT / folder).rglob('*')):
                if not source.is_file() or any(part in {'logs_train', 'assets', 'build', '__pycache__'} for part in source.relative_to(ROOT / folder).parts):
                    continue
                if source.suffix.lower() not in SOURCE_SUFFIXES and source.name not in {'Makefile', 'requirements.txt'}:
                    continue
                output.write(f'\n{"=" * 72}\nFILE: {source.relative_to(ROOT).as_posix()}\n{"=" * 72}\n')
                output.write(source.read_text(encoding='utf-8-sig'))
                output.write('\n')
        for book in workbooks:
            output.write(f'\n{"=" * 72}\nWORKBOOK: {book.name}\n{"=" * 72}\n')
            output.writelines(workbook_text(book))
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
