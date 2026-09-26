"""Group one PCB's fitted components into a practical picking list."""

from collections import defaultdict
from dataclasses import dataclass
import re
from html import escape


CATEGORIES=('电阻','电容','电感','IC','二极管/LED','晶体管','连接器','开关','保险丝','晶振','电池','声学器件','其他')


def natural_key(text):
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r'(\d+)',text))


def category_for(ref,footprint='',value=''):
    prefix=(re.match(r'[A-Za-z]+',ref) or [''])[0].upper()
    description=f'{footprint} {value}'.upper()
    if (prefix in {'J','P','CON'} or
            any(token in description for token in ('CONN-','CONNECTOR','HDR-','USB-C','TERMINAL'))):
        return '连接器'
    if prefix in {'SPK','BZ'} or 'SPEAKER' in description or 'BUZZER' in description:
        return '声学器件'
    if prefix.startswith('SW'): return '开关'
    if prefix.startswith('LED'): return '二极管/LED'
    if prefix.startswith(('BT','BAT')): return '电池'
    return {'R':'电阻','C':'电容','L':'电感','U':'IC','IC':'IC',
            'D':'二极管/LED','Q':'晶体管','F':'保险丝','S':'开关',
            'Y':'晶振','X':'晶振','B':'电池'}.get(prefix,'其他')


def _text(value):
    return ' '.join(str(value or '').split())


@dataclass(frozen=True)
class InventoryRow:
    category: str
    value: str
    footprint: str
    supplier_part: str
    sides: str
    mounting: str
    refs: tuple[str,...]

    @property
    def quantity(self):
        return len(self.refs)


def inventory_rows(board):
    """Exclude DNP/ignored parts; keep unresolved mounting visible for review."""
    groups=defaultdict(list)
    for component in board.components:
        if component.dnp or component.mounting=='ignore':
            continue
        category=category_for(component.ref,component.footprint,component.value)
        value=_text(component.value)
        footprint=_text(component.footprint)
        supplier=_text(component.properties.get('Supplier Part'))
        # Without a catalog code, identical printed values can hide different ratings.
        unidentified=component.uid if not supplier else ''
        key=(category,value.casefold(),footprint.casefold(),supplier.casefold(),unidentified)
        groups[key].append(component)
    rows=[]
    for members in groups.values():
        first=members[0]
        mounts={c.mounting for c in members}
        mount=('待确认' if 'unknown' in mounts else
               '贴装/插装' if len(mounts)>1 else
               '贴装' if 'smt' in mounts else '插装')
        rows.append(InventoryRow(category_for(first.ref,first.footprint,first.value),_text(first.value) or '未填写',
                                 _text(first.footprint) or '未填写',
                                 _text(first.properties.get('Supplier Part')),
                                 '顶/底' if len({c.side for c in members})>1 else
                                 '顶面' if first.side=='top' else '底面',
                                 mount,tuple(sorted((c.ref for c in members),key=natural_key))))
    order={name:i for i,name in enumerate(CATEGORIES)}
    rows.sort(key=lambda row:(order[row.category],natural_key(row.value),
                              natural_key(row.footprint),natural_key(row.supplier_part),
                              natural_key(row.refs[0])))
    return rows


def save_inventory_png(path, board_title, rows):
    """Render the complete list, including rows outside the on-screen viewport."""
    from pathlib import Path
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QTextDocument

    cells=[]
    previous=None
    for row in rows:
        if row.category!=previous:
            cells.append(f'<tr bgcolor="#dcecf2"><td colspan="7"><b>{escape(row.category)}</b></td></tr>')
            previous=row.category
        values=(row.value,row.footprint,row.supplier_part or '未填写',row.sides,
                row.mounting,str(row.quantity),'、'.join(row.refs))
        cells.append('<tr>'+''.join(f'<td>{escape(value)}</td>' for value in values)+'</tr>')
    html=(f'<h2>PCB 找料与焊接清单 · {escape(board_title)}</h2>'
          f'<p>共 {sum(row.quantity for row in rows)} 个待安装元件，{len(rows)} 种料件。'
          '不含工程标记不装配或手动忽略的元件；“待确认”表示安装类型需核对。</p>'
          '<table width="100%" border="1" cellspacing="0" cellpadding="6" '
          'style="border-color:#b8c9d2;font-size:11pt">'
          '<tr bgcolor="#23475e"><th width="160"><font color="white">规格值</font></th>'
          '<th width="200"><font color="white">封装</font></th>'
          '<th width="155"><font color="white">立创 C 编号</font></th>'
          '<th width="65"><font color="white">板面</font></th>'
          '<th width="95"><font color="white">安装</font></th>'
          '<th width="65"><font color="white">数量</font></th>'
          '<th width="360"><font color="white">位号</font></th></tr>'
          +''.join(cells)+'</table>')
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists(): QFontDatabase.addApplicationFont(str(font_path))
    document=QTextDocument(); document.setDefaultFont(QFont('Microsoft YaHei',10))
    document.setTextWidth(1100); document.setHtml(html)
    height=max(200,int(document.size().height()+32))
    if height>15000:
        raise ValueError('清单图片过高，无法保存为单张 PNG。')
    image=QImage(2264,2*height,QImage.Format.Format_RGB32)
    if image.isNull(): raise ValueError('无法分配清单图片内存。')
    image.fill(QColor('#ffffff'))
    painter=QPainter(image)
    painter.scale(2,2); painter.translate(16,16)
    document.drawContents(painter)
    painter.end()
    if not image.save(str(path),'PNG'): raise ValueError('PNG 图片保存失败。')
