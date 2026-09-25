"""Read-only EasyEDA Pro history reader. Never writes to the source database."""
from __future__ import annotations

import base64
import gzip
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ImportFailure(ValueError):
    pass


@dataclass
class Document:
    uuid: str
    kind: str
    header: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)
    deleted: bool = False

    def of_type(self, kind):
        return [(key[1], value) for key, value in self.records.items() if key[0] == kind]


def iter_records(text):
    """Decode JSON first so delimiters inside property strings stay intact."""
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in ' \r\n\t|':
            pos += 1
        if pos == len(text):
            return
        header, pos = decoder.raw_decode(text, pos)
        if not text.startswith('||', pos):
            raise ImportFailure('工程记录缺少分隔符，无法可靠读取。')
        pos += 2
        if pos == len(text) or text[pos] in '|\r\n':
            payload = None
        else:
            payload, pos = decoder.raw_decode(text, pos)
        yield header, payload


def read_project(path, progress=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise ImportFailure('找不到工程文件。')
    docs = {}
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')  # Keep all history reads on one database snapshot.
        projects = connection.execute('SELECT name, branch_uuid FROM projects').fetchall()
        if len(projects) != 1:
            raise ImportFailure('目前仅支持包含一个项目的工程容器。')
        name, branch = projects[0]
        if not re.fullmatch(r'[a-fA-F0-9]+', branch or ''):
            raise ImportFailure('无法识别当前分支。')
        table = 'project_history_' + branch
        histories = {r[0]: r[1:] for r in connection.execute(
            f'SELECT uuid,parent,key,snapshot FROM "{table}"')}
        tip = connection.execute('SELECT history_uuid FROM branches WHERE uuid=?', (branch,)).fetchone()
        if not tip or not tip[0]:
            raise ImportFailure('当前分支没有可识别的历史头。')
        chain, visited, cursor = [], set(), tip[0]
        while cursor:
            if cursor in visited or cursor not in histories:
                raise ImportFailure('历史链循环或缺失，拒绝使用不完整 PCB。')
            visited.add(cursor)
            parent, key, snapshot = histories[cursor]
            if snapshot:
                raise ImportFailure('此工程使用尚未支持的历史快照，请导出完整工程。')
            chain.append((cursor, key))
            cursor = parent
        row_counts=dict(connection.execute('SELECT history_uuid,COUNT(*) FROM history_data GROUP BY history_uuid'))
        total=sum(row_counts.get(history,0) for history,_ in chain)
        completed=0
        for history, key in reversed(chain):
            rows = connection.execute(
                'SELECT uuid,dataStr FROM history_data WHERE history_uuid=? ORDER BY id', (history,))
            for uuid, encoded in rows:
                prefix = re.match(r'[a-fA-F0-9]+', uuid)
                if not prefix:
                    raise ImportFailure('历史记录标识无效。')
                try:
                    raw = AESGCM(bytes.fromhex(key)).decrypt(
                        bytes.fromhex(prefix[0]), base64.b64decode(encoded, validate=True), None)
                    text = gzip.decompress(raw).decode('utf-8')
                except Exception as exc:
                    raise ImportFailure('工程历史校验或解码失败，无法继续。') from exc
                current = None
                for head, payload in iter_records(text):
                    kind = head['type']
                    if kind == 'EDIT_HEAD':
                        continue
                    if kind == 'DOCHEAD':
                        if not payload:
                            raise ImportFailure('文档头不完整。')
                        uid = payload['uuid']
                        current = docs.setdefault(uid, Document(uid, payload['docType']))
                        current.header.update(payload)
                        continue
                    if current is None:
                        raise ImportFailure('历史片段缺少文档归属。')
                    if kind == 'DELETE_DOC':
                        current.deleted = bool((payload or {}).get('isDelete', False))
                        continue
                    record_key = (kind, head.get('id', kind))
                    if payload is None:
                        current.records.pop(record_key, None)
                    else:
                        current.records[record_key] = payload
                completed+=1
                if progress and (completed==total or completed % max(1,total//100)==0):
                    progress(completed,total)
        return name, {uid: doc for uid, doc in docs.items() if not doc.deleted}
    except sqlite3.Error as exc:
        raise ImportFailure('不支持的 EasyEDA 工程数据库结构：' + str(exc)) from exc
    finally:
        connection.close()
