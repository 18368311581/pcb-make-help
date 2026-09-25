"""Shared layout, solid generation and paired export in millimetres."""
from __future__ import annotations

import csv
import json
import math
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .model import Board, Segment


@dataclass
class Parameters:
    base: float = 1.0
    wall: float = 1.0
    margin: float = 3.0
    pcb_thickness: float = 1.6
    stencil_thickness: float = 0.12
    pcb_gap: float = 0.2
    stencil_gap: float = 0.2
    wall_height: float = 1.72  # Above the top of the base, not above the table.
    gaps: list = field(default_factory=lambda: [[12.5,25.0],[37.5,25.0],[62.5,25.0],[87.5,25.0]])
    tray_base: float = 1.5
    pocket_depth: float = 1.0
    pocket_gap: float = 0.5
    min_wall: float = 1.0
    tray_margin: float = 5.0
    spacing: float = 1.0
    text_size: float = 1.8
    text_depth: float = 0.25
    labels: bool = True
    properties_on_labels: bool = False
    print_width: float = 256.0
    print_height: float = 256.0

    def validate(self, tool):
        if tool not in ('fixture','tray'): raise ValueError('未知工具。')
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if not math.isfinite(value) or value<0: raise ValueError(f'{key} 必须是有限非负数。')
        required=('base','wall','pcb_thickness','stencil_thickness','wall_height') if tool=='fixture' else ('tray_base','pocket_depth','min_wall','tray_margin')
        for key in required+('print_width','print_height'):
            if getattr(self,key)<=0: raise ValueError(f'{key} 必须大于 0。')
        if self.spacing<1: raise ValueError('布局间距倍率不能小于 1。')
        if tool=='fixture':
            if self.margin<max(self.pcb_gap,self.stencil_gap)+self.wall:
                raise ValueError('外围余量必须覆盖定位间隙和侧壁厚度。')
            if self.wall_height<=self.pcb_thickness:
                raise ValueError('侧壁必须高于 PCB，才能定位钢网。')
            if abs(self.stencil_gap-self.pcb_gap)>=self.wall:
                raise ValueError('上下定位壁必须有重叠，间隙差必须小于壁厚。')
            if not self.gaps: raise ValueError('至少保留一个操作缺口，避免形成封闭围墙。')
            for position,width in self.gaps:
                if not (math.isfinite(position) and 0<=position<100 and math.isfinite(width) and width>0):
                    raise ValueError('缺口位置应为 0～100%（不含100），宽度应大于0。')
        if self.labels and tool=='tray':
            if self.text_size<=0 or self.text_depth<=0 or self.text_depth>=self.tray_base:
                raise ValueError('文字尺寸/深度必须大于0，雕刻深度必须小于底厚。')


def board_polygon(board, side):
    if side not in ('top','bottom'): raise ValueError('未知板面。')
    p=Polygon(board.points())
    if not p.is_valid or p.area<=0: raise ValueError('板框自交或无有效面积。')
    cx=(p.bounds[0]+p.bounds[2])/2; cy=(p.bounds[1]+p.bounds[3])/2
    p=affinity.translate(p,-cx,-cy)
    return affinity.scale(p,xfact=-1,yfact=1,origin=(0,0)) if side=='bottom' else p


def selected(board,side):
    candidates=[c for c in board.components if c.side==side]
    unknown=[c.ref for c in candidates if c.mounting=='unknown']
    if unknown: raise ValueError('请确认安装类型：'+', '.join(unknown))
    items=[c for c in candidates if c.mounting=='smt']
    if not items: raise ValueError('此面无贴装元件。')
    return items


def pocket_polygon(board,c,p):
    """Tray coordinates after spacing and bottom mirroring; source PCB coordinates stay intact."""
    source=Polygon(board.points())
    cx=(source.bounds[0]+source.bounds[2])/2; cy=(source.bounds[1]+source.bounds[3])/2
    local=box(c.offset_x-c.width/2-p.pocket_gap,c.offset_y-c.height/2-p.pocket_gap,
              c.offset_x+c.width/2+p.pocket_gap,c.offset_y+c.height/2+p.pocket_gap)
    if c.side=='bottom': local=affinity.scale(local,xfact=-1,yfact=1,origin=(0,0))
    poly=affinity.rotate(local,c.angle,origin=(0,0))
    centre=poly.centroid
    poly=affinity.translate(poly,(c.x-cx)*p.spacing+centre.x*(p.spacing-1),
                            (c.y-cy)*p.spacing+centre.y*(p.spacing-1))
    if c.side=='bottom': poly=affinity.scale(poly,xfact=-1,yfact=1,origin=(0,0))
    return affinity.translate(poly,c.tray_offset_x,c.tray_offset_y)


def label_position(c,poly):
    return poly.centroid.x+c.label_offset_x,poly.bounds[3]+.45+c.label_offset_y


def label_size(c,p):
    return p.text_size if c.label_size_override is None else c.label_size_override


@lru_cache(maxsize=256)
def label_outline(text,size):
    """The exact XY contours used by CadQuery lettering, sampled for the flat view."""
    import cadquery as cq
    glyph=cq.Workplane('XY').text(text,size,.1,font='Arial',halign='center',
                                   valign='bottom',combine=False).val()
    if glyph.Volume()<=0: raise ValueError('字体没有生成可预览的笔画。')
    rings=[]
    for face in glyph.Faces():
        if face.geomType()!='PLANE' or abs(face.Center().z)>1e-6 or face.normalAt().z>-.9:
            continue
        for wire in face.Wires():
            points=[]
            for edge in wire.Edges():
                sampled=[edge.startPoint(),edge.endPoint()] if edge.geomType()=='LINE' else edge.sample(.005)[0]
                points.extend((point.x,point.y) for point in sampled[:-1])
            if points: rings.append(tuple(points))
    if not rings: raise ValueError('字体轮廓为空。')
    return tuple(rings)


def pocket_layout(board,side,p,preview=False):
    p.validate('tray')
    issues=[]
    if preview:
        candidates=[c for c in board.components if c.side==side]
        unknown=[c.ref for c in candidates if c.mounting=='unknown']
        if unknown: issues.append('安装类型待确认、未绘制凹槽：'+', '.join(unknown))
        items=[c for c in candidates if c.mounting=='smt']
        if not items: raise ValueError('此面尚无已确认的贴装元件。')
    else:
        items=selected(board,side)
    pockets=[]
    for c in items:
        values=(c.width,c.height,c.offset_x,c.offset_y,c.tray_offset_x,c.tray_offset_y,
                c.label_offset_x,c.label_offset_y)
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) or c.width<=0 or c.height<=0:
            issues.append(c.ref+' 缺少有效尺寸'); continue
        if not c.confirmed: issues.append(c.ref+' 尺寸待确认')
        size=label_size(c,p)
        if c.label_size_override is not None and (not isinstance(size,(int,float)) or not math.isfinite(size) or size<=0):
            issues.append(c.ref+' 标注字号必须为有限正数'); continue
        pockets.append((c,pocket_polygon(board,c,p)))
    conflicts=[]
    for i,(a,pa) in enumerate(pockets):
        for b,pb in pockets[i+1:]:
            if pa.distance(pb)<p.min_wall-1e-7:
                conflicts.append((a.ref,b.ref))
    if not pockets: raise ValueError('没有可生成的凹槽，请补充尺寸。')
    x0=min(poly.bounds[0] for _,poly in pockets)-p.tray_margin
    y0=min(poly.bounds[1] for _,poly in pockets)-p.tray_margin
    x1=max(poly.bounds[2] for _,poly in pockets)+p.tray_margin
    y1=max(poly.bounds[3] for _,poly in pockets)+p.tray_margin
    outer=box(x0,y0,x1,y1)
    if x1-x0>p.print_width or y1-y0>p.print_height:
        issues.append(f'备料盘 {x1-x0:.1f} × {y1-y0:.1f} mm 超出配置的打印范围。')
    return pockets,outer,conflicts,issues


def outline_wire(board,side):
    import cadquery as cq
    points=board.points(); cx=(min(x for x,y in points)+max(x for x,y in points))/2
    cy=(min(y for x,y in points)+max(y for x,y in points))/2
    def vec(pt): return cq.Vector((pt[0]-cx)*(-1 if side=='bottom' else 1),pt[1]-cy,0)
    edges=[]
    for s in board.outline:
        if abs(s.angle)<1e-8: edges.append(cq.Edge.makeLine(vec(s.start),vec(s.end)))
        else: edges.append(cq.Edge.makeThreePointArc(vec(s.start),vec(s.point(.5)),vec(s.end)))
    wire=cq.Wire.assembleEdges(edges)
    if not wire.IsClosed(): raise ValueError('建模板框不闭合。')
    return wire


def offset(wire,amount):
    if abs(amount)<1e-9: return wire
    result=wire.offset2D(amount,'arc')
    if len(result)!=1: raise ValueError('板框偏移产生多个轮廓，请检查狭窄区域和间隙。')
    return result[0]


def prism(wire,height,z=0):
    import cadquery as cq
    return cq.Solid.extrudeLinear(wire,[],cq.Vector(0,0,height)).translate((0,0,z))


def polygon_prism(poly,height,z=0):
    import cadquery as cq
    wire=cq.Wire.makePolygon([cq.Vector(x,y,0) for x,y in list(poly.exterior.coords)[:-1]],close=True)
    holes=[cq.Wire.makePolygon([cq.Vector(x,y,0) for x,y in list(ring.coords)[:-1]],close=True)
           for ring in poly.interiors]
    return cq.Solid.extrudeLinear(wire,holes,cq.Vector(0,0,height)).translate((0,0,z))


def gap_polygons(board,side,p):
    boundary=board_polygon(board,side).exterior
    result=[]
    for position,width in p.gaps:
        distance=boundary.length*position/100
        pt=boundary.interpolate(distance)
        before=boundary.interpolate((distance-.02)%boundary.length)
        after=boundary.interpolate((distance+.02)%boundary.length)
        angle=math.degrees(math.atan2(after.y-before.y,after.x-before.x))
        cut=box(-width/2,-(p.margin+p.wall+2)*2,width/2,(p.margin+p.wall+2)*2)
        result.append(affinity.translate(affinity.rotate(cut,angle,origin=(0,0)),pt.x,pt.y))
    return result


def build_fixture(board,side,p,progress=None):
    p.validate('fixture')
    if not any(c.side==side and c.mounting in ('smt','unknown') for c in board.components):
        raise ValueError('此面无贴装元件。')
    wire=outline_wire(board,side)
    if progress: progress(12,'板框检查完成，生成底板…')
    base=prism(offset(wire,p.margin),p.base)
    if progress: progress(27,'底板完成，生成分段侧壁…')
    walls=[]
    cuts=gap_polygons(board,side,p)
    steps=2*(len(cuts)+1); completed=0
    for layer,(gap,z,h) in enumerate([(p.pcb_gap,p.base,p.pcb_thickness),
                     (p.stencil_gap,p.base+p.pcb_thickness,p.wall_height-p.pcb_thickness)],1):
        ring=prism(offset(wire,gap+p.wall),h,z).cut(prism(offset(wire,gap),h+.02,z-.01))
        completed+=1
        if progress: progress(27+38*completed//steps,f'生成第 {layer}/2 层侧壁…')
        for cut in cuts:
            ring=ring.cut(polygon_prism(cut,h+.04,z-.02))
            completed+=1
            if progress: progress(27+38*completed//steps,f'切除第 {layer}/2 层操作缺口…')
        if not ring.Solids(): raise ValueError('缺口移除了全部侧壁，请减小宽度或减少缺口。')
        walls.append(ring)
    solid=base.fuse(*walls).clean()
    if progress: progress(75,'侧壁合并完成，检查实体…')
    check_solid(solid)
    bb=solid.BoundingBox()
    if bb.xlen>p.print_width or bb.ylen>p.print_height: raise ValueError('夹具超出配置的打印范围。')
    warnings=['按背面无元件、完整底面支撑生成。','定位间隙与平整度需试打印验证。']
    if p.wall_height>p.pcb_thickness+p.stencil_thickness+.001:
        warnings.append('侧壁高于钢网工作面，请确认缺口与刮刀方向。')
    if progress: progress(85,'夹具实体检查完成。')
    return solid,warnings


def check_solid(solid):
    if not solid.isValid() or len(solid.Solids())!=1 or solid.Volume()<=0:
        raise ValueError('几何验证失败：必须是一个有效、连续的实体。')


def verify_stl(path,expected_bounds,expected_volume,progress=None):
    data=Path(path).read_bytes()
    if len(data)<84: raise ValueError('STL 文件不完整。')
    count=struct.unpack_from('<I',data,80)[0]
    if not count or len(data)!=84+50*count: raise ValueError('STL 三角形数量与文件长度不符。')
    edges=Counter(); mins=[float('inf')]*3; maxs=[-float('inf')]*3; volume=0
    for i in range(count):
        if progress and i % max(1,count//20)==0: progress(i,count)
        values=struct.unpack_from('<12fH',data,84+50*i)
        vertices=[tuple(values[j:j+3]) for j in (3,6,9)]
        for vertex in vertices:
            for axis,v in enumerate(vertex):
                if not math.isfinite(v): raise ValueError('STL 包含非有限坐标。')
                mins[axis]=min(mins[axis],v); maxs[axis]=max(maxs[axis],v)
        a,b,c=vertices
        volume+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6
        quantized=[tuple(round(v,5) for v in vertex) for vertex in vertices]
        if len(set(quantized))<3: raise ValueError('STL 包含退化三角形。')
        for j in range(3): edges[tuple(sorted((quantized[j],quantized[(j+1)%3])))]+=1
    if any(n!=2 for n in edges.values()): raise ValueError('STL 网格不闭合或存在非流形边。')
    dims=[maxs[i]-mins[i] for i in range(3)]
    if max(abs(a-b) for a,b in zip(dims,expected_bounds))>.05:
        raise ValueError('STL 尺寸误差超过 0.05 mm。')
    if abs(abs(volume)-expected_volume)>max(.1,expected_volume*.005):
        raise ValueError('STL 网格体积与实体不符。')
    if progress: progress(count,count)
    return {'triangles':count,'watertight':True,'dimensions_mm':dims,'volume_mm3':abs(volume)}


def build_tray(board,side,p,progress=None):
    import cadquery as cq
    pockets,outer,conflicts,issues=pocket_layout(board,side,p)
    if issues: raise ValueError('\n'.join(issues[:18])+('\n……' if len(issues)>18 else ''))
    total=p.tray_base+p.pocket_depth
    labels=[]
    if p.labels:
        for i,(c,poly) in enumerate(pockets,1):
            content=c.tray_label.strip()
            if not content: continue
            try:
                glyph=cq.Workplane('XY').text(content,label_size(c,p),p.text_depth+.02,
                                                 font='Arial',halign='center',valign='bottom',combine=False).val()
                if glyph.Volume()<=0: raise ValueError('字体没有生成可雕刻的笔画')
                x,y=label_position(c,poly)
                glyph=glyph.translate((x,y,0)); bounds=glyph.BoundingBox()
                if not all(math.isfinite(v) for v in (bounds.xmin,bounds.xmax,bounds.ymin,bounds.ymax)):
                    raise ValueError('文字边界无效')
                labels.append((c,glyph))
            except Exception as exc:
                raise ValueError(f'{c.ref} 标注生成失败：{exc}') from exc
            if progress and (i==len(pockets) or i%max(1,len(pockets)//20)==0):
                progress(3+7*i//len(pockets),f'生成备料标注 {i}/{len(pockets)}…')
        if labels:
            x0,y0,x1,y1=outer.bounds
            for _,glyph in labels:
                bounds=glyph.BoundingBox()
                if bounds.xmin<x0: x0=bounds.xmin-p.tray_margin
                if bounds.ymin<y0: y0=bounds.ymin-p.tray_margin
                if bounds.xmax>x1: x1=bounds.xmax+p.tray_margin
                if bounds.ymax>y1: y1=bounds.ymax+p.tray_margin
            outer=box(x0,y0,x1,y1)
    x0,y0,x1,y1=outer.bounds
    if x1-x0>p.print_width or y1-y0>p.print_height:
        raise ValueError(f'备料盘 {x1-x0:.1f} × {y1-y0:.1f} mm 超出配置的打印范围。')
    if progress: progress(10,f'已检查 {len(pockets)} 个凹槽，生成盘体…')
    solid=polygon_prism(outer,total)
    if progress: progress(20,'盘体完成，合并相交凹槽…')
    merged=unary_union([poly for _,poly in pockets])
    if progress: progress(30,'凹槽轮廓合并完成，建立切削体…')
    regions=[merged] if merged.geom_type=='Polygon' else list(merged.geoms)
    tools=[]
    for i,poly in enumerate(regions,1):
        tools.append(polygon_prism(poly,p.pocket_depth+.02,p.tray_base))
        if progress: progress(30+10*i//len(regions),f'建立凹槽切削体 {i}/{len(regions)}…')
    solid=solid.cut(*tools).clean()
    if progress: progress(55,'凹槽切削完成，准备标注雕刻…')
    warnings=[]
    if conflicts:
        overlaps=sum(pa.intersects(pb) and pa.intersection(pb).area>1e-8
                     for i,(_,pa) in enumerate(pockets) for _,pb in pockets[i+1:])
        warnings.append(f'{len(conflicts)} 对凹槽相交或隔墙不足；其中 {overlaps} 对相交槽已形成连通凹槽。')
    if labels:
        cutters=[]
        for i,(c,glyph) in enumerate(labels,1):
            bounds=glyph.BoundingBox()
            footprint=box(bounds.xmin-.01,bounds.ymin-.01,bounds.xmax+.01,bounds.ymax+.01)
            pieces=[]
            for region,z in ((footprint.difference(merged),total-p.text_depth),
                             (footprint.intersection(merged),p.tray_base-p.text_depth)):
                polygons=[region] if region.geom_type=='Polygon' else [g for g in region.geoms if g.geom_type=='Polygon']
                for polygon in polygons:
                    if polygon.area<=1e-10: continue
                    mask=polygon_prism(polygon,total+2,-1)
                    cut=glyph.translate((0,0,z)).intersect(mask)
                    if cut.Volume()>1e-10: pieces.append(cut)
            if not pieces: raise ValueError(f'{c.ref} 标注没有落在可雕刻的盘体表面。')
            cutters.extend(pieces)
            if progress: progress(55+20*i//len(labels),f'准备备料标注 {i}/{len(labels)}…')
        before=solid.Volume()
        if progress: progress(76,f'雕刻 {len(labels)} 处备料标注…')
        solid=solid.cut(*cutters).clean()
        if before-solid.Volume()<=1e-9: raise ValueError('标注未能从盘体移除材料，请检查位置和字体。')
    if progress: progress(82,'标注完成，生成方向缺口…')
    # A physical notch identifies the +Y edge, also drawn in the flat preview.
    x0,y0,x1,y1=outer.bounds
    notch=Polygon([((x0+x1)/2-1,y1+.01),((x0+x1)/2+1,y1+.01),((x0+x1)/2,y1-1)])
    solid=solid.cut(polygon_prism(notch,total+.02,-.01)).clean()
    check_solid(solid)
    if progress: progress(85,'备料盘实体检查完成。')
    return solid,warnings


def build(board,side,tool,p,progress=None):
    if progress: progress(0,'生成夹具实体…' if tool=='fixture' else '计算备料盘凹槽…')
    return build_fixture(board,side,p,progress) if tool=='fixture' else build_tray(board,side,p,progress)


def export_pair(solid,directory,board,side,tool,p,source,source_hash,warnings,progress=None):
    import cadquery as cq
    if progress: progress(0,'检查待导出实体…')
    check_solid(solid)
    # Meshing attaches triangulations to OCCT faces; measure analytic geometry first.
    analytic=solid.copy(mesh=False)
    expected=analytic.BoundingBox()
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    title=re.sub(r'[^\w.-]+','_',board.title).strip('.') or 'PCB'
    stem=f'{title}_{tool}_{side}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output=root/stem
    output.mkdir(exist_ok=False)
    stp=output/(stem+'.stp'); stl=output/(stem+'.stl')
    if progress: progress(10,'写出 STEP 实体…')
    cq.exporters.export(analytic,str(stp),exportType='STEP')
    if progress: progress(30,'STEP 写出完成，回读检查…')
    read=cq.importers.importStep(str(stp)).val(); check_solid(read)
    actual=read.BoundingBox()
    if max(abs(getattr(expected,k)-getattr(actual,k)) for k in ('xlen','ylen','zlen'))>1e-4:
        raise ValueError('STEP 回读尺寸不一致。')
    if progress: progress(48,'STEP 回读通过，写出 STL…')
    cq.exporters.export(analytic,str(stl),exportType='STL',tolerance=.02,angularTolerance=.1)
    if progress: progress(65,'STL 写出完成，检查闭合性和尺寸…')
    mesh_report=verify_stl(stl,[expected.xlen,expected.ylen,expected.zlen],solid.Volume(),
                           lambda done,total: progress(65+30*done//total,f'检查 STL 三角形 {done}/{total}…') if progress else None)
    if progress: progress(97,'几何验证完成，写出报告与元件清单…')
    report={'schema':1,'units':'mm','source':str(source),'source_sha256':source_hash,
            'board':board.uid,'title':board.title,'side':side,'tool':tool,'parameters':asdict(p),
            'warnings':warnings,'dimensions_mm':[expected.xlen,expected.ylen,expected.zlen],
            'solid_volume_mm3':solid.Volume(),'step_readback_valid':True,
            'stl_validation':mesh_report,
            'components':[asdict(c) for c in board.components],
            'manual_checks_pending':['CATIA 导入及编辑','切片软件检查','实际打印与试配']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'元件清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['位号','板面','类型','属性','备料标注','封装','建议宽mm','建议高mm','尺寸已确认'])
        for c in board.components: writer.writerow([c.ref,c.side,c.mounting,c.value,c.tray_label,c.footprint,c.width,c.height,c.confirmed])
    if progress: progress(99,'报告与元件清单完成。')
    return output
