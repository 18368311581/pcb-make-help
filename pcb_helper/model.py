from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .eprj import ImportFailure, read_project

MIL = 0.0254


@dataclass
class Segment:
    start: tuple
    end: tuple
    angle: float = 0.0

    def point(self, t):
        x, y = self.start
        dx, dy = self.end[0] - x, self.end[1] - y
        if abs(self.angle) < 1e-8:
            return (x + dx * t, y + dy * t)
        theta = math.radians(self.angle)
        k = 0.5 / math.tan(theta / 2)
        cx, cy = x + dx / 2 - dy * k, y + dy / 2 + dx * k
        co, si = math.cos(theta * t), math.sin(theta * t)
        return (cx + (x-cx)*co - (y-cy)*si, cy + (x-cx)*si + (y-cy)*co)


def path_segments(path, scale=MIL):
    if not path:
        raise ImportFailure('空轮廓。')
    if path[0] == 'R':
        x,y,w,h,rot = path[1:6]
        # v3.2 uses R,x,y,w,h,rot,radius; newer documentation adds isCCW.
        r = path[7] if len(path)>7 else (path[6] if len(path)>6 else 0)
        if w<=0 or h<=0 or r<0 or r>min(w,h)/2:
            raise ImportFailure('矩形轮廓尺寸无效。')
        points = [(x+r,y),(x+w-r,y),(x+w,y-r),(x+w,y-h+r),
                  (x+w-r,y-h),(x+r,y-h),(x,y-h+r),(x,y-r)]
        co,si = math.cos(math.radians(rot)),math.sin(math.radians(rot))
        points=[((x+(a-x)*co-(b-y)*si)*scale,(y+(a-x)*si+(b-y)*co)*scale) for a,b in points]
        return [Segment(points[i],points[(i+1)%8],-90 if i%2 else 0)
                for i in range(8) if math.dist(points[i],points[(i+1)%8])>1e-9]
    if path[0] == 'CIRCLE':
        x,y,r=path[1:4]
        if r<=0: raise ImportFailure('圆轮廓半径无效。')
        return [Segment(((x+r)*scale,y*scale),((x-r)*scale,y*scale),180),
                Segment(((x-r)*scale,y*scale),((x+r)*scale,y*scale),180)]
    if not isinstance(path[0],(int,float)):
        raise ImportFailure('暂不支持的轮廓指令：'+str(path[0]))
    current=(path[0]*scale,path[1]*scale)
    pos,mode,result=2,'L',[]
    while pos<len(path):
        if isinstance(path[pos],str):
            mode=path[pos]; pos+=1
        if mode=='L':
            end=(path[pos]*scale,path[pos+1]*scale); pos+=2; angle=0
        elif mode in ('ARC','CARC'):
            angle=path[pos]; end=(path[pos+1]*scale,path[pos+2]*scale); pos+=3
        else:
            raise ImportFailure('暂不支持的轮廓曲线：'+mode)
        if math.dist(current,end)>1e-9:
            result.append(Segment(current,end,angle))
        current=end
    return result


def sampled(segments):
    points=[]
    for s in segments:
        steps=max(1,math.ceil(abs(s.angle)/3))
        points.extend(s.point(i/steps) for i in range(steps))
    if segments: points.append(segments[-1].end)
    return points


@dataclass
class Component:
    uid: str
    ref: str
    x: float
    y: float
    angle: float
    side: str
    footprint: str
    value: str
    mounting: str
    reason: str
    width: float = 0
    height: float = 0
    offset_x: float = 0
    offset_y: float = 0
    size_source: str = '缺少尺寸'
    confirmed: bool = False
    dnp: bool = False
    properties: dict = field(default_factory=dict)
    tray_label: str = ''
    tray_offset_x: float = 0.0
    tray_offset_y: float = 0.0
    label_offset_x: float = 0.0
    label_offset_y: float = 0.0
    label_size_override: float | None = None


def suggest_tray_label(component):
    """Use short, source-derived procurement cues; unknown/IC labels stay blank."""
    if component.mounting!='smt' or component.dnp:
        return ''
    kind=re.match(r'[A-Za-z]+',component.ref)
    kind=kind.group().upper() if kind else ''
    if kind not in {'R','C','L','D','Q','F'}:
        return ''
    value=' '.join(component.value.split()).strip()
    if not value or value.casefold() in {'n/a','unknown','none','resistor','capacitor','inductor','diode'}:
        return ''
    if value.casefold() in {component.ref.casefold(),component.footprint.casefold()}:
        return ''
    if kind in {'R','C','L'}:
        return value if len(value)<=16 and any(ch.isdigit() for ch in value) else ''
    return value if len(value)<=16 else ''


@dataclass
class Board:
    uid: str
    title: str
    outline: list
    components: list
    holes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def points(self):
        return sampled(self.outline)


def _component_size(footprint):
    if not footprint: return 0,0,0,0,'缺少封装'
    points=[]
    for _,v in footprint.of_type('POLY'):
        if v.get('layerId')==48:
            try: points.extend(sampled(path_segments(v['path'])))
            except (ImportFailure,IndexError,TypeError): pass
    for _,v in footprint.of_type('LINE'):
        if v.get('layerId')==48:
            points.extend([(v['startX']*MIL,v['startY']*MIL),(v['endX']*MIL,v['endY']*MIL)])
    body=bool(points)
    # Pads are only a conservative *proposal*, never silently confirmed as a body.
    for _,v in footprint.of_type('PAD'):
        p=v.get('defaultPad',{})
        w,h=p.get('width',0)*MIL,p.get('height',0)*MIL
        a=math.radians((v.get('padAngle') or 0)+(v.get('relativeAngle') or 0))
        for x,y in [(-w/2,-h/2),(-w/2,h/2),(w/2,h/2),(w/2,-h/2)]:
            points.append((v['centerX']*MIL+x*math.cos(a)-y*math.sin(a),
                           v['centerY']*MIL+x*math.sin(a)+y*math.cos(a)))
    if not points: return 0,0,0,0,'缺少尺寸'
    xs,ys=zip(*points); x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
    return x1-x0,y1-y0,(x0+x1)/2,(y0+y1)/2,('器件形状＋焊盘包络（待确认）' if body else '仅焊盘包络（待确认）')


def _board_outline(doc):
    groups=[]
    for _,v in doc.of_type('POLY'):
        if v.get('layerId')==11:
            groups.append(path_segments(v['path']))
    singles=[]
    for kind in ('LINE','ARC','CARC'):
        for _,v in doc.of_type(kind):
            if v.get('layerId')==11:
                singles.append(Segment((v['startX']*MIL,v['startY']*MIL),
                                       (v['endX']*MIL,v['endY']*MIL),v.get('angle',0)))
    if singles: groups.extend([[s] for s in singles])
    if not groups: raise ImportFailure('未找到支持的板框。')
    segments=groups.pop(0)
    while groups:
        endpoint=segments[-1].end
        for i,g in enumerate(groups):
            if math.dist(endpoint,g[0].start)<1e-4:
                segments.extend(groups.pop(i)); break
            if math.dist(endpoint,g[-1].end)<1e-4:
                segments.extend([Segment(s.end,s.start,-s.angle) for s in reversed(groups.pop(i))]); break
        else:
            raise ImportFailure('板框不连续或含多个独立轮廓，暂不支持此输入；不会用矩形代替。')
    if math.dist(segments[0].start,segments[-1].end)>1e-4:
        raise ImportFailure('板框没有闭合。')
    # Some EasyEDA revisions repeat the exact same closed path. One circuit is
    # the physical outline; traversing it twice would make the polygon invalid.
    for period in range(1,len(segments)//2+1):
        if len(segments)%period or math.dist(segments[0].start,segments[period-1].end)>1e-4:
            continue
        if all(math.dist(s.start,segments[i%period].start)<1e-6 and
               math.dist(s.end,segments[i%period].end)<1e-6 and
               abs(s.angle-segments[i%period].angle)<1e-6
               for i,s in enumerate(segments)):
            segments=segments[:period]
            break
    return segments


def load_boards(path, progress=None):
    project,docs=read_project(path, progress)
    boards=[]
    for doc in docs.values():
        if doc.kind!='PCB': continue
        attrs={}
        for _,v in doc.of_type('ATTR'):
            attrs.setdefault(v.get('parentId'),{})[v['key']]=v.get('value','')
        components=[]
        for uid,c in doc.of_type('COMPONENT'):
            a=attrs.get(uid,{})
            fp=docs.get(a.get('Footprint')); dev=docs.get(a.get('Device'))
            meta=dev.of_type('META')[0][1] if dev and dev.of_type('META') else {}
            props=dict(meta.get('attributes',{})); props.update(c.get('attrs',{})); props.update(a)
            title=fp.of_type('META')[0][1].get('title','') if fp and fp.of_type('META') else ''
            pads=fp.of_type('PAD') if fp else []
            holes=sum(bool(p.get('hole')) for _,p in pads)
            surface=sum(p.get('layerId') in (1,2) and not p.get('hole') for _,p in pads)
            mount='unknown'; reason='无可靠安装类型'
            explicit=str(props.get('Mounting Type',props.get('Mount Type',''))).upper()
            if explicit in ('SMD','SMT','SURFACE MOUNT'): mount,reason='smt','工程安装类型'
            elif explicit in ('TH','THT','THROUGH HOLE'): mount,reason='tht','工程安装类型'
            elif surface and not holes: mount,reason='smt','封装仅含表贴焊盘'
            elif holes and not surface: mount,reason='tht','封装仅含通孔焊盘'
            elif holes and surface: reason='混合封装，需人工确认'
            dnp=str(props.get('DNP','')).lower() in ('true','yes','1') or str(props.get('Add into BOM','')).lower()=='no'
            if dnp: mount,reason='ignore','工程标记不装配/不加入 BOM，请核对'
            if c.get('layerId') not in (1,2): raise ImportFailure('元件板面无法识别：'+a.get('Designator',uid))
            w,h,ox,oy,source=_component_size(fp)
            value=str(props.get('Value') or props.get('Manufacturer Part') or meta.get('title',''))
            component=Component(uid,a.get('Designator',uid),c['x']*MIL,c['y']*MIL,
                c.get('angle',0),'top' if c['layerId']==1 else 'bottom',title,value,mount,reason,
                w,h,ox,oy,source,False,dnp,props)
            component.tray_label=suggest_tray_label(component)
            components.append(component)
        components.sort(key=lambda c:[int(s) if s.isdigit() else s for s in re.split(r'(\d+)',c.ref)])
        holes=[v for _,v in doc.of_type('PAD') if v.get('hole')]
        holes.extend(v for _,v in doc.of_type('REGION') if v.get('regionType')=='BOARD_CUTOUT')
        title=dict(doc.of_type('META')).get('META',{}).get('title',doc.uuid)
        boards.append(Board(doc.uuid,title,_board_outline(doc),components,holes))
    if not boards: raise ImportFailure('工程中没有有效 PCB。')
    return project,boards


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
