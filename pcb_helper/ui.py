from __future__ import annotations

import json
import math
import os
import traceback
import numpy as np
from dataclasses import asdict, fields
from pathlib import Path
from shapely.geometry import Point, box

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF, QLineF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QBrush, QFont, QPainterPath, QFontDatabase, QWheelEvent
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QLabel,QPushButton,QComboBox,QLineEdit,QFileDialog,QMessageBox,QSplitter,QTabWidget,
    QFormLayout,QDoubleSpinBox,QCheckBox,QRadioButton,QButtonGroup,QScrollArea,
    QTableWidget,QTableWidgetItem,QHeaderView,QGraphicsView,QGraphicsScene,QAbstractItemView,
    QDialog,QProgressBar,QTextBrowser)

import vtkmodules.vtkRenderingOpenGL2  # Register the OpenGL renderer.
import vtkmodules.vtkInteractionStyle
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkRenderingCore import vtkActor, vtkLight, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from vtkmodules.vtkIOImage import vtkPNGWriter

from .model import load_boards, fingerprint, suggest_tray_label
from .inventory import inventory_rows, save_inventory_png
from .geometry import Parameters, board_polygon, pocket_layout, pocket_polygon, label_position, label_size, label_outline, gap_polygons, build, export_pair

APP_DIR=Path(__file__).resolve().parent
ROOT=APP_DIR.parent  # Development samples and tests only.
DATA_DIR=APP_DIR/'output'
SETTINGS=DATA_DIR/'last_session.json'
RECENT_CONFIG=DATA_DIR/'recent_config.json'
MOUNTS={'smt':'贴装','tht':'插装','ignore':'忽略','unknown':'待确认'}
COL={'ref':0,'side':1,'mount':2,'value':3,'label':4,'pocket_x':5,'pocket_y':6,
     'label_x':7,'label_y':8,'font':9,'width':10,'height':11,'confirmed':12,'source':13,'footprint':14}
NUMERIC_COLUMNS={5,6,7,8,9,10,11}
SIZE_SOURCE_SHORT={
    '器件形状＋焊盘包络（待确认）':'外形+焊盘',
    '仅焊盘包络（待确认）':'仅焊盘',
    '手工输入':'手工',
    '缺少尺寸':'缺尺寸',
    '缺少封装':'缺封装',
}
LABELS={
 'base':'底板厚度','wall':'侧壁厚度','margin':'外围余量','pcb_thickness':'PCB 厚度',
 'stencil_thickness':'钢网厚度','pcb_gap':'PCB 单边间隙','stencil_gap':'钢网单边间隙',
 'wall_height':'壁高（底板上表面起）','tray_base':'盘底厚度','pocket_depth':'凹槽深度',
 'pocket_gap':'槽单边取放间隙','min_wall':'最小隔墙','tray_margin':'盘外边距',
 'spacing':'槽中心间距倍率','text_size':'文字字号','text_depth':'雕刻深度',
 'print_width':'打印范围 X','print_height':'打印范围 Y'}
CONFIG_KEYS=(*LABELS,'labels','gaps')


def configure_fonts(app):
    # Explicit registration also makes Windows fonts available to offscreen Qt.
    import os
    directory=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    family='Microsoft YaHei'
    for name in ('msyh.ttc','msyhbd.ttc','arial.ttf'):
        path=directory/name
        if path.exists():
            ident=QFontDatabase.addApplicationFont(str(path))
            if name=='msyh.ttc' and ident>=0:
                families=QFontDatabase.applicationFontFamilies(ident)
                if families: family=families[0]
    app.setFont(QFont(family,10))


def configure_appearance(app):
    app.setStyleSheet('''
        QMainWindow { background: #f7f9fb; }
        QPushButton { background: #ffffff; color: #263b50; border: 1px solid #b9c8d3;
                      border-radius: 5px; padding: 5px 9px; min-height: 23px; }
        QPushButton:hover { background: #edf7f8; border-color: #58a3ad; }
        QPushButton:pressed { background: #dceff1; }
        QPushButton:disabled { background: #f0f2f4; color: #929da6; border-color: #d7dfe5; }
        QPushButton#primaryAction { background: #16889a; color: #ffffff; border-color: #13798a; font-weight: bold; }
        QPushButton#primaryAction:hover { background: #117688; }
        QComboBox, QLineEdit { background: #ffffff; color: #263b50; border: 1px solid #b9c8d3;
                      border-radius: 4px; padding: 3px 5px; min-height: 22px; }
        QComboBox:focus, QLineEdit:focus { border-color: #16889a; }
        QComboBox QAbstractItemView { background: #ffffff; color: #263b50;
                     selection-background-color: #d9eff2; selection-color: #17364a;
                     border: 1px solid #b9c8d3; outline: 0; }
        QComboBox QAbstractItemView::item:selected { background: #d9eff2; color: #17364a; }
        QTabWidget::pane { border: 1px solid #bac9d4; background: #ffffff; }
        QTabBar::tab { background: #edf2f5; border: 1px solid #bac9d4;
                       padding: 6px 14px; margin-right: 2px; }
        QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; color: #126f80; }
    ''')


class Worker(QThread):
    result=Signal(object)
    failed=Signal(str)
    progress=Signal(int,str)
    def __init__(self,fn): super().__init__(); self.fn=fn
    def run(self):
        try: self.result.emit(self.fn())
        except Exception as exc:
            log=DATA_DIR/'errors.log'; log.parent.mkdir(exist_ok=True)
            with log.open('a',encoding='utf-8') as stream: stream.write(traceback.format_exc()+'\n')
            self.failed.emit(str(exc))


class SortableItem(QTableWidgetItem):
    def __lt__(self,other):
        if self.column() in NUMERIC_COLUMNS:
            try: return float(self.text())<float(other.text())
            except ValueError: pass
        if self.column()==0:
            import re
            def parts(value):
                return tuple((1,int(s)) if s.isdigit() else (0,s.casefold())
                             for s in re.split(r'(\d+)',value))
            return parts(self.text())<parts(other.text())
        return self.text().casefold()<other.text().casefold()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self,event):
        viewport=self.parentWidget()
        if viewport:
            forwarded=QWheelEvent(QPointF(self.mapTo(viewport,event.position().toPoint())),
                                  event.globalPosition(),event.pixelDelta(),event.angleDelta(),
                                  event.buttons(),event.modifiers(),event.phase(),event.inverted())
            QApplication.sendEvent(viewport,forwarded)
            event.accept()
        else:
            event.ignore()


class FlatPreview(QGraphicsView):
    def __init__(self):
        super().__init__(); self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor('#f3f6fa'))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.pan_anchor=None; self.pan_button=None
        self.mode='pan'; self.on_move=None
        self.pocket_targets=[]; self.label_targets=[]; self.drag=None; self.connector=None
    def clear_layout(self):
        self.scene().clear(); self.pocket_targets=[]; self.label_targets=[]
        self.drag=None; self.connector=None; self.pan_anchor=None; self.pan_button=None
    def set_mode(self,mode):
        if self.drag is not None:
            for item,origin in self.drag[3]: item.setPos(origin)
        self.mode=mode; self.pan_anchor=None; self.pan_button=None
        self.drag=None; self.remove_connector(); self.unsetCursor()
    def polygon(self,poly,color,fill=None):
        if poly.is_empty: return
        if poly.geom_type=='MultiPolygon':
            for p in poly.geoms: self.polygon(p,color,fill)
            return
        path=QPainterPath(); path.setFillRule(Qt.FillRule.OddEvenFill)
        for ring in [poly.exterior,*poly.interiors]:
            path.addPolygon(QPolygonF([QPointF(x,-y) for x,y in ring.coords])); path.closeSubpath()
        return self.scene().addPath(path,QPen(QColor(color),.18),QBrush(QColor(fill)) if fill else QBrush(Qt.BrushStyle.NoBrush))
    def label(self,x,y,text,size=1.5,color='#263b50'):
        path=QPainterPath(); path.setFillRule(Qt.FillRule.OddEvenFill)
        for ring in label_outline(text,size):
            path.addPolygon(QPolygonF([QPointF(px,-py) for px,py in ring])); path.closeSubpath()
        item=self.scene().addPath(path,QPen(Qt.PenStyle.NoPen),QBrush(QColor(color)))
        item.setPos(x,-y)
        return item
    def remove_connector(self):
        if self.connector is not None:
            self.scene().removeItem(self.connector); self.connector=None
    def update_connector(self,uid):
        label=next((target for target in self.label_targets if target[0]==uid),None)
        pocket=next((target for target in self.pocket_targets if target[0]==uid),None)
        if label is None or pocket is None: return
        centre=pocket[1].centroid
        pocket_scene=QPointF(centre.x,-centre.y)+pocket[2].pos()
        label_scene=label[3].sceneBoundingRect().center()
        if self.connector is None:
            pen=QPen(QColor('#db6725')); pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(11); pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap); pen.setDashPattern([1.5,1.5])
            self.connector=self.scene().addLine(QLineF(pocket_scene,label_scene),pen)
            self.connector.setZValue(100)
        else: self.connector.setLine(QLineF(pocket_scene,label_scene))
    def add_pocket_target(self,uid,poly,item):
        self.pocket_targets.append((uid,poly,item))
    def add_label_target(self,uid,x,y,item,pocket_item):
        self.label_targets.append((uid,x,y,item,pocket_item))
    def hit_target(self,scene_point):
        x,y=scene_point.x(),-scene_point.y()
        if self.mode=='pocket':
            matches=[(poly.centroid.distance(Point(x,y)),uid,item)
                     for uid,poly,item in self.pocket_targets if poly.covers(Point(x,y))]
            if matches:
                _,uid,item=min(matches,key=lambda entry:(entry[0],entry[1]))
                label_items=[target[3] for target in self.label_targets if target[0]==uid]
                return uid,[item,*label_items]
        if self.mode=='label':
            matches=[(math.hypot(x-lx,y-ly),uid,item)
                     for uid,lx,ly,item,_ in self.label_targets if item.sceneBoundingRect().contains(scene_point)]
            if matches:
                _,uid,item=min(matches,key=lambda entry:(entry[0],entry[1]))
                return uid,[item]
        return None
    def fit(self):
        rect=self.scene().itemsBoundingRect().adjusted(-4,-4,4,4)
        self.setSceneRect(rect.adjusted(-rect.width()*100,-rect.height()*100,rect.width()*100,rect.height()*100))
        self.fitInView(rect,Qt.AspectRatioMode.KeepAspectRatio)
    def wheelEvent(self,event):
        cursor=event.position().toPoint()
        before=self.mapToScene(cursor)
        factor=1.15**(event.angleDelta().y()/120)
        self.scale(factor,factor)
        after=self.mapToScene(cursor)
        center=self.mapToScene(self.viewport().rect().center())
        self.centerOn(center+before-after)
        event.accept()
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.RightButton or (event.button()==Qt.MouseButton.LeftButton and self.mode=='pan'):
            self.pan_button=event.button()
            self.pan_anchor=self.mapToScene(event.position().toPoint())
            self.setCursor(Qt.CursorShape.ClosedHandCursor); event.accept(); return
        if event.button()==Qt.MouseButton.LeftButton and self.mode in ('pocket','label'):
            scene_pos=self.mapToScene(event.position().toPoint())
            target=self.hit_target(scene_pos)
            if target:
                uid,items=target
                self.drag=(uid,scene_pos,items,[(item,item.pos()) for item in items])
                if self.mode=='label': self.update_connector(uid)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.pan_button is not None:
            current=self.mapToScene(event.position().toPoint())
            center=self.mapToScene(self.viewport().rect().center())
            next_center=center+self.pan_anchor-current
            rect=self.sceneRect()
            self.setSceneRect(rect.united(QRectF(next_center.x()-rect.width()/4,
                                                  next_center.y()-rect.height()/4,
                                                  rect.width()/2,rect.height()/2)))
            self.centerOn(next_center); event.accept()
        elif self.drag is not None:
            _,start,_,origins=self.drag
            delta=self.mapToScene(event.position().toPoint())-start
            for item,origin in origins: item.setPos(origin+delta)
            if self.mode=='label': self.update_connector(self.drag[0])
            event.accept()
        else: super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if event.button()==self.pan_button:
            self.pan_button=None; self.pan_anchor=None; self.unsetCursor(); event.accept(); return
        if event.button()==Qt.MouseButton.LeftButton:
            if self.drag is not None:
                uid,start,_,_=self.drag
                delta=self.mapToScene(event.position().toPoint())-start
                self.drag=None; self.remove_connector(); self.unsetCursor()
                if self.on_move and delta.manhattanLength()>1e-7:
                    self.on_move(uid,self.mode,delta.x(),-delta.y())
                event.accept(); return
            self.unsetCursor(); event.accept()
        else: super().mouseReleaseEvent(event)
    def contextMenuEvent(self,event): event.accept()


def zoom_parallel_at_cursor(renderer,x,y,factor):
    """Keep the world point under an off-centre cursor fixed during parallel zoom."""
    camera=renderer.GetActiveCamera()
    renderer.SetWorldPoint(*camera.GetFocalPoint(),1)
    renderer.WorldToDisplay()
    depth=renderer.GetDisplayPoint()[2]
    def at_cursor():
        renderer.SetDisplayPoint(x,y,depth)
        renderer.DisplayToWorld()
        point=renderer.GetWorldPoint()
        return np.asarray(point[:3],dtype=float)/point[3]
    before=at_cursor()
    camera.SetParallelScale(max(camera.GetParallelScale()*factor,1e-6))
    after=at_cursor()
    shift=before-after
    camera.SetPosition(*(np.asarray(camera.GetPosition())+shift))
    camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint())+shift))


class CadViewport(QVTKRenderWindowInteractor):
    def __init__(self,parent=None):
        super().__init__(parent); self.last=None; self.drag_button=None; self.renderer=vtkRenderer()
        self.renderer.SetBackground(.79,.84,.89)
        self.renderer.SetBackground2(.94,.96,.98); self.renderer.GradientBackgroundOn()
        self.renderer.AutomaticLightCreationOff()
        for position,color,intensity in [((-1,-1,2.5),(1,.98,.93),.85),
                                         ((1,.7,1.5),(.82,.90,1),.42)]:
            light=vtkLight(); light.SetLightTypeToSceneLight(); light.SetPosition(*position)
            light.SetFocalPoint(0,0,0); light.PositionalOff()
            light.SetColor(*color); light.SetIntensity(intensity)
            self.renderer.AddLight(light)
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.Initialize()
        axes=vtkAxesActor(); self.axes=vtkOrientationMarkerWidget()
        self.axes.SetOrientationMarker(axes); self.axes.SetInteractor(self.GetRenderWindow().GetInteractor())
        self.axes.SetViewport(0.01,0.01,0.16,0.20); self.axes.SetEnabled(1); self.axes.InteractiveOff()
    def mousePressEvent(self,event):
        if self.drag_button is None and event.button() in (Qt.MouseButton.LeftButton,Qt.MouseButton.RightButton):
            self.last=event.position(); self.drag_button=event.button()
        event.accept()
    def mouseMoveEvent(self,event):
        if self.last is None:
            event.accept(); return
        delta=event.position()-self.last; self.last=event.position()
        camera=self.renderer.GetActiveCamera()
        if self.drag_button==Qt.MouseButton.RightButton:
            camera.Azimuth(-delta.x()*.4); camera.Elevation(delta.y()*.4); camera.OrthogonalizeViewUp()
        else:
            height=max(1,self.height()); scale=camera.GetParallelScale()*2/height
            right=np.asarray(camera.GetDirectionOfProjection()); up=np.asarray(camera.GetViewUp())
            right=np.cross(right,up); right/=max(np.linalg.norm(right),1e-12)
            shift=(-delta.x()*right+delta.y()*up)*scale
            camera.SetPosition(*(np.asarray(camera.GetPosition())+shift))
            camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint())+shift))
        self.renderer.ResetCameraClippingRange(); self.GetRenderWindow().Render(); event.accept()
    def mouseReleaseEvent(self,event):
        if event.button()==self.drag_button: self.last=None; self.drag_button=None
        event.accept()
    def mouseDoubleClickEvent(self,event): event.accept()
    def keyPressEvent(self,event): event.accept()
    def keyReleaseEvent(self,event): event.accept()
    def contextMenuEvent(self,event): event.accept()
    def wheelEvent(self,event):
        width,height=self.GetRenderWindow().GetSize()
        if width>0 and height>0 and event.angleDelta().y():
            x=event.position().x()*width/max(self.width(),1)
            y=height-event.position().y()*height/max(self.height(),1)
            zoom_parallel_at_cursor(self.renderer,x,y,1.12**(-event.angleDelta().y()/120))
        self.renderer.ResetCameraClippingRange(); self.GetRenderWindow().Render(); event.accept()


class SolidPreview(QWidget):
    def __init__(self):
        super().__init__(); self.setMinimumSize(400,300); layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.view=CadViewport(self); layout.addWidget(self.view)
        self.actor=None; self.edge_actor=None; self.has_mesh=False
    def clear_mesh(self,render=True):
        if self.actor: self.view.renderer.RemoveActor(self.actor)
        if self.edge_actor: self.view.renderer.RemoveActor(self.edge_actor)
        self.actor=None; self.edge_actor=None; self.has_mesh=False
        if render: self.view.GetRenderWindow().Render()
    def set_mesh(self,mesh):
        vertices,faces=mesh
        points=np.asarray(vertices,dtype=np.float32); triangles=np.asarray(faces,dtype=np.int64)
        poly=vtkPolyData(); poly.SetPoints(vtkPoints())
        poly.GetPoints().SetData(numpy_to_vtk(points,deep=True))
        cells=np.empty((len(triangles),4),dtype=np.int64); cells[:,0]=3; cells[:,1:]=triangles
        faces_vtk=vtkCellArray(); faces_vtk.SetCells(len(triangles),numpy_to_vtkIdTypeArray(cells.ravel(),deep=True)); poly.SetPolys(faces_vtk)
        mapper=vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.ScalarVisibilityOff()
        self.clear_mesh(render=False); actor=vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(.35,.56,.70); actor.GetProperty().SetAmbient(.14)
        actor.GetProperty().SetDiffuse(.78); actor.GetProperty().SetSpecular(.22)
        actor.GetProperty().SetSpecularPower(24)
        self.view.renderer.AddActor(actor); self.actor=actor; self.has_mesh=True

        edges=vtkFeatureEdges(); edges.SetInputData(poly)
        edges.BoundaryEdgesOff(); edges.NonManifoldEdgesOff(); edges.ManifoldEdgesOff()
        edges.FeatureEdgesOn(); edges.SetFeatureAngle(25)
        edge_mapper=vtkPolyDataMapper(); edge_mapper.SetInputConnection(edges.GetOutputPort())
        edge_mapper.ScalarVisibilityOff(); edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        edge_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-2,-2)
        edge_actor=vtkActor(); edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetColor(.09,.20,.29)
        edge_actor.GetProperty().SetOpacity(.85); edge_actor.GetProperty().SetLineWidth(1.2)
        edge_actor.GetProperty().LightingOff()
        self.view.renderer.AddActor(edge_actor); self.edge_actor=edge_actor
        camera=self.view.renderer.GetActiveCamera(); camera.ParallelProjectionOn()
        x0,x1,y0,y1,z0,z1=poly.GetBounds()
        cx,cy,cz=(x0+x1)/2,(y0+y1)/2,(z0+z1)/2
        span=max(x1-x0,y1-y0,z1-z0,1)
        camera.SetFocalPoint(cx,cy,cz)
        camera.SetPosition(cx+.20*span,cy-.30*span,cz+2.5*span)
        camera.SetViewUp(0,1,0)
        self.view.renderer.ResetCamera()
        camera.OrthogonalizeViewUp()
        forward=np.asarray(camera.GetDirectionOfProjection())
        up=np.asarray(camera.GetViewUp())
        right=np.cross(forward,up); right/=np.linalg.norm(right)
        up=np.cross(right,forward); up/=np.linalg.norm(up)
        corners=np.asarray([(x,y,z) for x in (x0,x1) for y in (y0,y1) for z in (z0,z1)])
        relative=corners-np.asarray(camera.GetFocalPoint())
        half_width=np.max(np.abs(relative@right))
        half_height=np.max(np.abs(relative@up))
        aspect=max(self.view.width(),1)/max(self.view.height(),1)
        camera.SetParallelScale(max(half_height,half_width/aspect,0.01)*1.12)
        self.view.renderer.ResetCameraClippingRange()
        self.view.GetRenderWindow().Render()
    def save_image(self,path):
        capture=vtkWindowToImageFilter(); capture.SetInput(self.view.GetRenderWindow())
        capture.ReadFrontBufferOff(); capture.Update()
        writer=vtkPNGWriter(); writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort()); writer.Write()


class BusyDialog(QDialog):
    def __init__(self,parent,title):
        super().__init__(parent); self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        self.setFixedWidth(390)
        layout=QVBoxLayout(self); self.message=QLabel(title); layout.addWidget(self.message)
        self.bar=QProgressBar(); self.bar.setRange(0,100); self.bar.setValue(0); layout.addWidget(self.bar)
    def update_progress(self,percent,message):
        self.message.setText(message)
        self.bar.setValue(max(0,min(percent,99)) if percent<100 else 100)
    def reject(self): pass
    def closeEvent(self,event): event.ignore()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('PCB 贴装助手'); self.resize(1450,940)
        DATA_DIR.mkdir(exist_ok=True)
        self.boards=[]; self.source=None; self.source_hash=''; self.solid=None; self.warnings=[]
        self.inventory_rows=[]
        self.worker=None; self.loading=False; self.side_state={'fixture':'top','tray':'top'}; self.current_tool='fixture'
        self.current_config_path=None; self.saved_parameters=None; self.config_dirty=False
        self.list_collapsed=False
        root=QWidget(); self.setCentralWidget(root); layout=QVBoxLayout(root)
        layout.setContentsMargins(12,10,12,8); layout.setSpacing(8)
        bar=QHBoxLayout(); self.format=QComboBox(); self.format.addItems(['EasyEDA 专业版','KiCad · 暂未支持','Altium · 暂未支持'])
        for i in (1,2): self.format.model().item(i).setEnabled(False)
        bar.addWidget(self.format); self.open_button=QPushButton('导入工程'); bar.addWidget(self.open_button)
        self.board_box=QComboBox(); self.board_box.setMinimumWidth(180); bar.addWidget(self.board_box)
        self.source_label=QLabel('尚未导入'); bar.addWidget(self.source_label,1)
        self.help_button=QPushButton('使用帮助'); bar.addWidget(self.help_button); layout.addLayout(bar)
        self.split=QSplitter(); layout.addWidget(self.split,1)
        left=QWidget(); left_layout=QVBoxLayout(left); left.setMinimumWidth(310); left.setMaximumWidth(420)
        self.tool=QComboBox(); self.tool.addItem('锡膏印刷定位夹具','fixture'); self.tool.addItem('元件备料盘','tray'); self.tool.addItem('找料与焊接清单','inventory'); left_layout.addWidget(self.tool)
        self.inventory_note=QLabel('按种类、规格值、封装与立创 C 编号整理所选 PCB 的顶底两面元件。\n\n切换到备料盘可修改元件安装类型；返回清单后会重新统计。')
        self.inventory_note.setWordWrap(True); left_layout.addWidget(self.inventory_note)
        self.inventory_spacer=QWidget(); left_layout.addWidget(self.inventory_spacer,1)
        self.side_panel=QWidget(); sidebar=QHBoxLayout(self.side_panel); sidebar.setContentsMargins(0,0,0,0)
        self.top=QRadioButton('顶面'); self.bottom=QRadioButton('底面'); self.top.setChecked(True)
        group=QButtonGroup(self); group.addButton(self.top); group.addButton(self.bottom); sidebar.addWidget(self.top); sidebar.addWidget(self.bottom); left_layout.addWidget(self.side_panel)
        self.side_hint=QLabel('导入后判断板面可用性'); self.side_hint.setWordWrap(True); left_layout.addWidget(self.side_hint)
        self.parameter_scroll=QScrollArea(); self.parameter_scroll.setWidgetResizable(True); panel=QWidget(); self.form=QFormLayout(panel); self.parameter_scroll.setWidget(panel); left_layout.addWidget(self.parameter_scroll,1)
        self.print_panel=QWidget(); self.print_form=QFormLayout(self.print_panel); left_layout.addWidget(self.print_panel)
        self.controls={}; defaults=Parameters()
        for key,label in LABELS.items():
            spin=QDoubleSpinBox(); spin.setRange(0,10000); spin.setDecimals(3); spin.setSingleStep(.1); spin.setValue(getattr(defaults,key)); spin.setSuffix(' 倍' if key=='spacing' else ' mm')
            if key=='spacing': spin.setRange(1,30)
            self.controls[key]=spin
            target=self.print_form if key in ('print_width','print_height') else self.form
            target.addRow(label,spin); spin.valueChanged.connect(self.parameter_changed)
        self.labels=QCheckBox('雕刻备料标注'); self.labels.setChecked(True); self.form.addRow(self.labels)
        self.labels.toggled.connect(self.parameter_changed)
        self.gap_label=QLabel('缺口：沿板框周长的位置 % / 宽度 mm'); self.form.addRow(self.gap_label)
        self.gaps=QTableWidget(0,2); self.gaps.setHorizontalHeaderLabels(['位置 %','宽度 mm']); self.gaps.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.gaps.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); self.form.addRow(self.gaps)
        gap_bar=QWidget(); gb=QHBoxLayout(gap_bar); gb.setContentsMargins(0,0,0,0)
        add=QPushButton('添加缺口'); remove=QPushButton('移除选中'); gb.addWidget(add); gb.addWidget(remove); self.form.addRow(gap_bar); self.gap_bar=gap_bar
        add.clicked.connect(lambda:self.add_gap(0,20)); remove.clicked.connect(self.remove_gap); self.gaps.cellChanged.connect(self.parameter_changed)
        self.set_gaps(defaults.gaps)
        self.reset_button=QPushButton('恢复默认'); self.reset_button.clicked.connect(self.reset_parameters); left_layout.addWidget(self.reset_button)
        self.config_label=QLabel('配置：未命名'); left_layout.addWidget(self.config_label)
        self.config_panel=QWidget(); configs=QHBoxLayout(self.config_panel); configs.setContentsMargins(0,0,0,0)
        save=QPushButton('保存'); save_as=QPushButton('另存为'); restore=QPushButton('打开配置')
        configs.addWidget(save); configs.addWidget(save_as); configs.addWidget(restore); left_layout.addWidget(self.config_panel)
        save.clicked.connect(self.save_dialog); save_as.clicked.connect(self.save_as_dialog); restore.clicked.connect(self.restore_dialog)
        self.generate=QPushButton('生成实体并预览'); self.generate.setObjectName('primaryAction'); self.generate.setMinimumHeight(40); left_layout.addWidget(self.generate)
        self.export=QPushButton('导出 STL + STP'); self.export.setEnabled(False); left_layout.addWidget(self.export)
        self.split.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); rl.setContentsMargins(0,0,0,0)
        self.right_split=QSplitter(Qt.Orientation.Vertical)
        self.right_split.setChildrenCollapsible(False); self.right_split.setHandleWidth(8)
        self.right_split.setStyleSheet('QSplitter::handle:vertical { background:#dbe3ea; border-top:1px solid #afbdc9; border-bottom:1px solid #afbdc9; }')
        preview_pane=QWidget(); preview_layout=QVBoxLayout(preview_pane); preview_layout.setContentsMargins(0,0,0,0)
        self.edit_bar=QWidget(); edit_layout=QHBoxLayout(self.edit_bar); edit_layout.setContentsMargins(4,0,4,0)
        edit_layout.addWidget(QLabel('二维操作：'))
        self.edit_group=QButtonGroup(self)
        for mode,title in (('pan','平移视图'),('pocket','移动凹槽'),('label','移动标注')):
            button=QRadioButton(title); self.edit_group.addButton(button); edit_layout.addWidget(button)
            button.toggled.connect(lambda checked,m=mode: self.flat.set_mode(m) if checked else None)
            if mode=='pan': self.pan_button=button
        edit_layout.addStretch(1)
        self.toggle_list=QPushButton('收起元件列表'); edit_layout.addWidget(self.toggle_list)
        self.toggle_list.clicked.connect(lambda:self.set_list_collapsed(not self.list_collapsed))
        preview_layout.addWidget(self.edit_bar)
        self.tabs=QTabWidget(); self.flat=FlatPreview(); self.three=SolidPreview(); self.tabs.addTab(self.flat,'布局 / 冲突'); self.tabs.addTab(self.three,'三维实体（未生成）')
        self.flat.on_move=self.move_preview_item
        self.pan_button.setChecked(True)
        self.tabs.setTabEnabled(1,False); preview_layout.addWidget(self.tabs,1)
        self.right_split.addWidget(preview_pane)
        self.list_pane=QWidget(); list_layout=QVBoxLayout(self.list_pane); list_layout.setContentsMargins(0,0,0,0)
        self.table_bar=QWidget(); table_bar=QVBoxLayout(self.table_bar); table_bar.setContentsMargins(0,0,0,0)
        actions=QHBoxLayout(); actions.addWidget(QLabel('元件 · 双击编辑位置、字号与尺寸；选中多行后确认建议尺寸'),1)
        auto_labels=QPushButton('自动生成标注'); auto_labels.setToolTip('按工程属性重算当前 PCB 的标注；会覆盖已有手工修改。')
        confirm=QPushButton('确认选中尺寸'); actions.addWidget(auto_labels); actions.addWidget(confirm); table_bar.addLayout(actions)
        filters=QHBoxLayout(); self.search=QLineEdit(); self.search.setPlaceholderText('筛选位号、属性、备料标注、封装…')
        self.filter_side=QComboBox(); self.filter_side.addItem('全部板面','all'); self.filter_side.addItem('顶面','top'); self.filter_side.addItem('底面','bottom')
        self.filter_mount=QComboBox(); self.filter_mount.addItem('全部类型','all')
        for key,label in MOUNTS.items(): self.filter_mount.addItem(label,key)
        filters.addWidget(self.search,1); filters.addWidget(self.filter_side); filters.addWidget(self.filter_mount); table_bar.addLayout(filters)
        list_layout.addWidget(self.table_bar)
        self.table=QTableWidget(0,15); self.table.setHorizontalHeaderLabels(['位号','板面','类型','属性','备料标注',
            '槽 X mm','槽 Y mm','标注 X mm','标注 Y mm','标注字号 mm','宽 mm','高 mm','已确认','尺寸依据','封装'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        for col,width in enumerate((70,58,76,110,120,85,85,90,90,98,74,74,68,115,140)): self.table.setColumnWidth(col,width)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0,Qt.SortOrder.AscendingOrder)
        list_layout.addWidget(self.table,1)
        self.right_split.addWidget(self.list_pane)
        self.right_split.setStretchFactor(0,3); self.right_split.setStretchFactor(1,2)
        self.right_split.setSizes([520,330])
        self.right_split.handle(1).setToolTip('拖动以调整元件列表高度')
        rl.addWidget(self.right_split,1)
        self.inventory_pane=QWidget(); inventory_layout=QVBoxLayout(self.inventory_pane)
        inventory_layout.setContentsMargins(0,0,0,0)
        self.inventory_heading=QLabel('找料与焊接清单'); self.inventory_heading.setObjectName('inventoryHeading')
        inventory_layout.addWidget(self.inventory_heading)
        self.inventory_table=QTableWidget(0,8)
        self.inventory_table.setHorizontalHeaderLabels(['种类','规格值','封装','立创 C 编号','板面','安装','数量','位号'])
        self.inventory_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.inventory_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.inventory_table.setAlternatingRowColors(True)
        self.inventory_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.inventory_table.horizontalHeader().setSectionResizeMode(7,QHeaderView.ResizeMode.Stretch)
        for column,width in enumerate((95,150,165,145,70,80,65)):
            self.inventory_table.setColumnWidth(column,width)
        inventory_layout.addWidget(self.inventory_table,1)
        self.inventory_pane.hide(); rl.addWidget(self.inventory_pane,1)
        self.split.addWidget(right); self.split.setSizes([340,1060])
        self.status=QLabel('请导入 EasyEDA 专业版 .eprj2。默认尺寸为试验起点，打印后需试配。'); self.status.setWordWrap(True); self.status.setMinimumHeight(42); layout.addWidget(self.status)
        self.open_button.clicked.connect(self.open_dialog); self.help_button.clicked.connect(self.show_help)
        self.board_box.currentIndexChanged.connect(self.board_changed); self.tool.currentIndexChanged.connect(self.tool_changed)
        self.top.toggled.connect(self.side_changed); self.bottom.toggled.connect(self.side_changed)
        self.table.itemChanged.connect(self.item_changed); confirm.clicked.connect(self.confirm_selected)
        auto_labels.clicked.connect(self.auto_generate_labels)
        self.search.textChanged.connect(self.filter_table); self.filter_side.currentIndexChanged.connect(self.filter_table)
        self.filter_mount.currentIndexChanged.connect(self.filter_table)
        self.generate.clicked.connect(self.generate_model); self.export.clicked.connect(self.export_model)
        self.timer=QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.refresh)
        self.tool_changed()
        self.saved_parameters=self.config_values(); self.update_config_title()
        QTimer.singleShot(0,self.restore_recent_config)

    def error(self,message):
        self.status.setText(message); QMessageBox.warning(self,'需要处理',message)
    def make_help_dialog(self):
        dialog=QDialog(self); dialog.setWindowTitle('使用帮助 · PCB 贴装助手'); dialog.resize(680,600)
        layout=QVBoxLayout(dialog)
        browser=QTextBrowser(); browser.setOpenExternalLinks(False)
        browser.setHtml('''
            <h2>PCB 贴装助手</h2>
            <p style="white-space: pre-wrap;">V1.0   MadeBy ChatGPT&江沛宸   In 20260925</p>
            <p>本地读取 EasyEDA 专业版 <code>.eprj2</code> 工程，生成锡膏印刷定位夹具、元件备料盘，
            或整理找料与焊接清单。两种打印工具可导出 STL 与实体 STEP；清单可导出完整表格 PNG。
            KiCad 和 Altium 导入暂未支持。</p>
            <h3>开始使用</h3>
            <ol>
              <li>点击“导入工程”，选定工程；如果工程包含多块 PCB，再从顶部列表选定一块。</li>
              <li>在左侧选择夹具或备料盘及顶面／底面，检查并调整尺寸。灰色板面表示没有可用的贴装元件。</li>
              <li>备料盘请先核对元件类型和宽高；待确认尺寸或安装类型应在元件表中处理。
                  备料标注可自动生成，也可逐项编辑、清空或拖动。</li>
              <li>检查二维布局和底部提示，然后点击“生成实体并预览”。修改尺寸或元件后，需要重新生成。</li>
              <li>点击“导出 STL + STP”并选择目录。程序会建立独立文件夹，包含模型、报告和元件清单。</li>
              <li>找料与焊接清单统计所选 PCB 两面的待安装元件，按大类、规格值、封装和立创 C 编号归并，
                  列出数量与位号。工程标记不装配或手动忽略的元件不计入；安装类型待确认的元件仍会标出。
                  点击“导出清单 PNG”保存完整表格图片。</li>
            </ol>
            <h3>视图与配置</h3>
            <p>二维视图：滚轮以鼠标位置缩放，右键拖动平移；备料盘可切换为左键移动凹槽或标注。
            三维视图：左键平移、右键旋转、滚轮以鼠标位置缩放。</p>
            <p>“保存”写入当前配置，“另存为”创建新配置，“打开配置”切换配置。
            配置只记录左侧参数，不包含 PCB 工程及元件表修改。下次启动会打开上次使用的配置。</p>
            <h3>输出与使用边界</h3>
            <p>STL 按毫米用于切片；STP 是可在 CAD 中继续修改的实体，不含 CATIA 原生特征历史。
            夹具只用于印刷锡膏，不随 PCB 上加热台。打印范围是软件中设置的 X/Y 可用尺寸，
            具体配合间隙和打印效果仍需用实际材料与设备检查。</p>
        ''')
        layout.addWidget(browser)
        close=QPushButton('关闭'); close.clicked.connect(dialog.accept); layout.addWidget(close,0,Qt.AlignmentFlag.AlignRight)
        return dialog
    def show_help(self): self.make_help_dialog().exec()
    def board(self):
        i=self.board_box.currentIndex(); return self.boards[i] if 0<=i<len(self.boards) else None
    def side(self): return 'bottom' if self.bottom.isChecked() else 'top'
    def params(self):
        values={key:control.value() for key,control in self.controls.items()}
        values.update(labels=self.labels.isChecked(),properties_on_labels=False,
                      gaps=[[float(self.gaps.item(r,c).text()) for c in range(2)] for r in range(self.gaps.rowCount())])
        return Parameters(**values)
    def config_values(self):
        values=asdict(self.params())
        return {key:values[key] for key in CONFIG_KEYS}
    def run_job(self,fn,done,title='正在处理'):
        if self.worker and self.worker.isRunning(): return
        self.centralWidget().setEnabled(False); self.status.setText('正在处理，请稍候……')
        self.busy=BusyDialog(self,title)
        worker=Worker(lambda:fn(lambda percent,message:worker.progress.emit(percent,message))); self.worker=worker
        worker.progress.connect(self.busy.update_progress)
        def succeeded(value):
            try:
                done(value); self.busy.update_progress(100,'完成')
                QTimer.singleShot(300,self.busy.hide)
            except Exception as exc:
                self.busy.hide(); self.error(str(exc))
        def failed(message): self.busy.hide(); self.error(message)
        worker.result.connect(succeeded); worker.failed.connect(failed)
        worker.finished.connect(lambda:self.centralWidget().setEnabled(True)); worker.start(); self.busy.show()
    def open_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'选择 EasyEDA 工程',str(Path.home()),'EasyEDA 专业版 (*.eprj2)')
        if path: self.load(path)
    def load(self,path):
        path=Path(path)
        def work(progress):
            progress(0,'核对工程文件…')
            digest=fingerprint(path)
            progress(2,'还原工程历史记录… 0%')
            project,boards=load_boards(path,lambda done,total:progress(2+94*done//total,f'还原工程历史记录… {done}/{total}'))
            progress(97,'核对工程一致性…')
            if fingerprint(path)!=digest:
                raise ValueError('工程在读取期间发生变化，请保存工程后重新导入。')
            progress(99,'工程读取完成，准备界面…')
            return project,boards,digest
        def done(result):
            project,boards,digest=result; self.loading=True; self.source=path; self.source_hash=digest; self.boards=boards
            self.board_box.clear(); self.board_box.addItems([b.title for b in boards]); self.board_box.setCurrentIndex(0 if len(boards)==1 else -1)
            self.source_label.setText(path.name); self.source_label.setToolTip(str(path)); self.loading=False
            self.board_changed()
        self.run_job(work,done,'导入 EasyEDA 工程')
    def board_changed(self):
        if self.loading: return
        self.flat_needs_fit=True
        self.populate_table(); self.changed()
    def populate_table(self):
        self.loading=True; self.table.setSortingEnabled(False); self.table.setRowCount(0); board=self.board()
        if board:
            self.table.setRowCount(len(board.components))
            p=self.params()
            for row,c in enumerate(board.components):
                coords=self.component_positions(c,p)
                vals=[c.ref,'顶面' if c.side=='top' else '底面',MOUNTS[c.mounting],c.value,
                      c.tray_label,*coords,f'{c.width:.3f}',f'{c.height:.3f}','',
                      SIZE_SOURCE_SHORT.get(c.size_source,c.size_source),c.footprint]
                for col,text in enumerate(vals):
                    item=SortableItem(text); item.setData(Qt.ItemDataRole.UserRole,c.uid)
                    if col not in (4,5,6,7,8,9,10,11): item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col==COL['confirmed']:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable); item.setCheckState(Qt.CheckState.Checked if c.confirmed else Qt.CheckState.Unchecked)
                    if col in (COL['source'],COL['footprint']): item.setToolTip(c.size_source if col==COL['source'] else c.footprint)
                    if col==COL['font']: item.setToolTip('留空表示使用左侧全局文字字号')
                    self.table.setItem(row,col,item)
                combo=NoWheelComboBox()
                for key,label in MOUNTS.items(): combo.addItem(label,key)
                combo.setCurrentIndex(list(MOUNTS).index(c.mounting)); combo.setToolTip(c.reason)
                combo.currentIndexChanged.connect(lambda _,uid=c.uid,w=combo:self.mount_changed(uid,w.currentData()))
                self.table.setCellWidget(row,2,combo)
        self.table.setSortingEnabled(True)
        self.loading=False
        self.filter_table()
    def component_positions(self,c,p):
        if c.mounting!='smt' or c.width<=0 or c.height<=0:
            return ['','','','','' if c.label_size_override is None else f'{c.label_size_override:.3f}']
        poly=pocket_polygon(self.board(),c,p); x,y=label_position(c,poly)
        return [f'{poly.centroid.x:.3f}',f'{poly.centroid.y:.3f}',f'{x:.3f}',f'{y:.3f}',
                '' if c.label_size_override is None else f'{c.label_size_override:.3f}']
    def sync_position_cells(self,p):
        self.loading=True; self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            c=self.component_at_row(row)
            if c:
                for col,value in enumerate(self.component_positions(c,p),COL['pocket_x']):
                    item=self.table.item(row,col)
                    if item and item.text()!=value: item.setText(value)
        self.table.setSortingEnabled(True); self.loading=False
    def component_by_uid(self,uid):
        board=self.board()
        return next((c for c in board.components if c.uid==uid),None) if board else None
    def component_at_row(self,row):
        item=self.table.item(row,0)
        return self.component_by_uid(item.data(Qt.ItemDataRole.UserRole)) if item else None
    def filter_table(self,*args):
        if not hasattr(self,'table'): return
        query=self.search.text().strip().casefold()
        side=self.filter_side.currentData(); mount=self.filter_mount.currentData()
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            c=self.component_at_row(row)
            haystack=' '.join((c.ref,c.value,c.tray_label,c.footprint,c.size_source)).casefold() if c else ''
            self.table.setRowHidden(row,not (c and (side=='all' or c.side==side)
                                              and (mount=='all' or c.mounting==mount)
                                              and (not query or query in haystack)))
    def mount_changed(self,uid,value):
        if self.loading: return
        c=self.component_by_uid(uid)
        if not c: return
        c.mounting=value
        for row in range(self.table.rowCount()):
            if self.component_at_row(row) is c:
                self.loading=True; self.table.item(row,2).setText(MOUNTS[value]); self.loading=False
                break
        self.filter_table(); self.changed()
    def item_changed(self,item):
        if self.loading or not self.board(): return
        c=self.component_by_uid(item.data(Qt.ItemDataRole.UserRole))
        if not c: return
        col=item.column()
        try:
            if col in (COL['width'],COL['height']):
                value=float(item.text())
                if not math.isfinite(value) or value<=0: raise ValueError('宽高必须是有限正数。')
                setattr(c,'width' if col==COL['width'] else 'height',value); c.size_source='手工输入'; c.confirmed=False
                self.loading=True
                self.table.item(item.row(),COL['confirmed']).setCheckState(Qt.CheckState.Unchecked)
                self.table.item(item.row(),COL['source']).setText(SIZE_SOURCE_SHORT.get(c.size_source,c.size_source))
                self.table.item(item.row(),COL['source']).setToolTip(c.size_source)
                self.loading=False
            elif col in (COL['pocket_x'],COL['pocket_y'],COL['label_x'],COL['label_y']):
                if c.mounting!='smt' or c.width<=0 or c.height<=0: raise ValueError('此元件没有可调整的贴装凹槽。')
                value=float(item.text())
                if not math.isfinite(value): raise ValueError('坐标必须是有限数字。')
                poly=pocket_polygon(self.board(),c,self.params())
                lx,ly=label_position(c,poly)
                current={COL['pocket_x']:poly.centroid.x,COL['pocket_y']:poly.centroid.y,
                         COL['label_x']:lx,COL['label_y']:ly}[col]
                offset={COL['pocket_x']:'tray_offset_x',COL['pocket_y']:'tray_offset_y',
                        COL['label_x']:'label_offset_x',COL['label_y']:'label_offset_y'}[col]
                setattr(c,offset,getattr(c,offset)+value-current)
            elif col==COL['font']:
                raw=item.text().strip()
                if raw:
                    value=float(raw)
                    if not math.isfinite(value) or value<=0: raise ValueError('标注字号必须是有限正数。')
                    c.label_size_override=value
                else: c.label_size_override=None
            elif col==COL['label']: c.tray_label=item.text().strip()
            elif col==COL['confirmed']: c.confirmed=item.checkState()==Qt.CheckState.Checked
            else: return
            if col in (COL['pocket_x'],COL['pocket_y'],COL['label_x'],COL['label_y'],COL['font'],COL['width'],COL['height']):
                self.sync_position_cells(self.params())
            self.filter_table()
            self.changed()
        except ValueError as exc: self.error(str(exc)); self.populate_table()
    def auto_generate_labels(self):
        board=self.board()
        if not board: return
        for c in board.components: c.tray_label=suggest_tray_label(c)
        self.populate_table(); self.changed()
    def confirm_selected(self):
        if not self.board(): return
        rows={index.row() for index in self.table.selectedIndexes()}
        if not rows: self.error('请先选择需要确认的元件行。'); return
        for row in rows:
            c=self.component_at_row(row)
            if c.width>0 and c.height>0: c.confirmed=True
        self.populate_table(); self.changed()
    def add_gap(self,pos,width):
        self.gaps.blockSignals(True); row=self.gaps.rowCount(); self.gaps.insertRow(row)
        for col,value in enumerate((pos,width)): self.gaps.setItem(row,col,QTableWidgetItem(str(value)))
        self.gaps.blockSignals(False); self.resize_gaps(); self.parameter_changed()
    def remove_gap(self):
        row=self.gaps.currentRow()
        if row>=0: self.gaps.removeRow(row); self.resize_gaps(); self.parameter_changed()
    def set_gaps(self,gaps):
        self.gaps.blockSignals(True); self.gaps.setRowCount(0)
        for pos,width in gaps: self.add_gap(pos,width)
        self.gaps.blockSignals(False)
        self.resize_gaps()
    def resize_gaps(self):
        height=self.gaps.horizontalHeader().height()+2*self.gaps.frameWidth()
        height+=sum(self.gaps.rowHeight(row) for row in range(self.gaps.rowCount()))
        self.gaps.setFixedHeight(height+2)
    def set_list_collapsed(self,collapsed):
        self.list_collapsed=bool(collapsed)
        self.list_pane.setVisible(self.current_tool=='tray' and not self.list_collapsed)
        self.toggle_list.setText('展开元件列表' if self.list_collapsed else '收起元件列表')
    def tool_changed(self):
        tool=self.tool.currentData(); self.current_tool=tool
        if tool in self.side_state:
            self.loading=True; self.top.setChecked(self.side_state[tool]=='top'); self.bottom.setChecked(self.side_state[tool]=='bottom'); self.loading=False
        inventory=tool=='inventory'
        for widget in (self.side_panel,self.side_hint,self.parameter_scroll,self.print_panel,
                       self.reset_button,self.config_label,self.config_panel):
            widget.setVisible(not inventory)
        self.inventory_note.setVisible(inventory); self.inventory_spacer.setVisible(inventory)
        self.right_split.setVisible(not inventory); self.inventory_pane.setVisible(inventory)
        self.generate.setText('重新整理清单' if inventory else '生成实体并预览')
        self.export.setText('导出清单 PNG' if inventory else '导出 STL + STP')
        fixture={'base','wall','margin','pcb_thickness','stencil_thickness','pcb_gap','stencil_gap','wall_height'}
        shared={'print_width','print_height'}
        for key,control in self.controls.items():
            if key in shared: continue
            show=tool!='inventory' and ((key in fixture)==(tool=='fixture'))
            self.form.setRowVisible(control,show)
        for w in (self.gaps,self.gap_label,self.gap_bar): self.form.setRowVisible(w,tool=='fixture')
        self.edit_bar.setVisible(tool=='tray')
        self.list_pane.setVisible(tool=='tray' and not self.list_collapsed)
        if tool!='tray': self.pan_button.setChecked(True)
        self.form.setRowVisible(self.labels,tool=='tray')
        self.form.invalidate(); self.form.activate(); self.changed()
    def side_changed(self):
        if not self.loading:
            self.side_state[self.current_tool]=self.side(); self.changed()
    def move_preview_item(self,uid,mode,dx,dy):
        if self.current_tool!='tray': return
        c=self.component_by_uid(uid)
        if not c: return
        if mode=='pocket':
            c.tray_offset_x+=dx; c.tray_offset_y+=dy
        elif mode=='label':
            c.label_offset_x+=dx; c.label_offset_y+=dy
        else: return
        self.sync_position_cells(self.params()); self.changed()
    def changed(self,*args):
        if self.loading: return
        self.solid=None
        if hasattr(self,'export'): self.export.setEnabled(False)
        if hasattr(self,'three'): self.three.clear_mesh()
        if hasattr(self,'tabs'):
            if self.tabs.currentIndex()==1: self.tabs.setCurrentIndex(0)
            self.tabs.setTabEnabled(1,False); self.tabs.setTabText(1,'三维实体（未生成）')
        if hasattr(self,'timer'): self.timer.start(150)
    def parameter_changed(self,*args):
        if self.loading: return
        self.changed()
        if self.saved_parameters is not None:
            try: self.config_dirty=self.config_values()!=self.saved_parameters
            except (ValueError,TypeError): self.config_dirty=True
            self.update_config_title()
    def populate_inventory(self,board):
        rows=inventory_rows(board); self.inventory_rows=rows
        table=self.inventory_table
        table.setSortingEnabled(False); table.setRowCount(len(rows))
        previous=None
        for index,row in enumerate(rows):
            values=(row.category,row.value,row.footprint,row.supplier_part or '未填写',
                    row.sides,row.mounting,str(row.quantity),'、'.join(row.refs))
            for column,value in enumerate(values):
                item=SortableItem(value)
                item.setToolTip(value)
                if column==6: item.setData(Qt.ItemDataRole.UserRole,row.quantity)
                if row.category!=previous:
                    item.setBackground(QColor('#e2f0f3'))
                    font=item.font(); font.setBold(True); item.setFont(font)
                table.setItem(index,column,item)
            previous=row.category
        count=sum(row.quantity for row in rows)
        missing=sum(row.quantity for row in rows if not row.supplier_part)
        unknown=sum(row.quantity for row in rows if row.mounting=='待确认')
        self.inventory_heading.setText(f'{board.title} · {count} 个元件 / {len(rows)} 种料件')
        notes=[]
        if missing: notes.append(f'{missing} 个元件缺少立创 C 编号')
        if unknown: notes.append(f'{unknown} 个元件安装类型待确认')
        self.status.setText('找料清单已整理；包含顶面与底面的贴装、插装元件。'
                            +(' '+ '；'.join(notes) if notes else ''))
        self.generate.setEnabled(True); self.export.setEnabled(True)
    def refresh(self):
        board=self.board(); self.flat.clear_layout()
        if not board:
            self.inventory_rows=[]; self.inventory_table.setRowCount(0)
            self.inventory_heading.setText('找料与焊接清单')
            self.status.setText('请选择 PCB。'); self.generate.setEnabled(False); self.export.setEnabled(False); return
        if self.current_tool=='inventory':
            self.populate_inventory(board)
            return
        summary=[]; available={}
        self.loading=True
        for side,radio,label in [('top',self.top,'顶面'),('bottom',self.bottom,'底面')]:
            items=[c for c in board.components if c.side==side]
            n=sum(c.mounting=='smt' for c in items); u=sum(c.mounting=='unknown' for c in items)
            available[side]=bool(n or u); radio.setEnabled(available[side]); radio.setToolTip('安装类型待确认' if u else ('' if n else '此面无贴装元件'))
            summary.append(f'{label}：{n} 贴装'+(f'，{u} 待确认' if u else ''))
        if not available[self.side()] and available['top' if self.side()=='bottom' else 'bottom']:
            self.top.setChecked(available['top']); self.bottom.setChecked(not available['top'])
        self.side_state[self.current_tool]=self.side(); self.loading=False; self.side_hint.setText(' / '.join(summary))
        try:
            p=self.params(); p.validate(self.current_tool); outline=board_polygon(board,self.side())
            issues=[]
            if self.current_tool=='tray':
                pockets,outer,conflicts,issues=pocket_layout(board,self.side(),p,preview=True)
                self.flat.polygon(outline,'#8a9cad')
                bad={r for pair in conflicts for r in pair}
                x0,y0,x1,y1=outer.bounds
                for c,poly in pockets:
                    item=self.flat.polygon(poly,'#c94343' if c.ref in bad else '#168994','#ffdcdc' if c.ref in bad else '#d6f2ed')
                    self.flat.add_pocket_target(c.uid,poly,item)
                    if p.labels and c.tray_label.strip():
                        lx,ly=label_position(c,poly)
                        label_item=self.flat.label(lx,ly,c.tray_label.strip(),label_size(c,p))
                        self.flat.add_label_target(c.uid,lx,ly,label_item,item)
                        bounds=label_item.sceneBoundingRect()
                        if bounds.left()<x0: x0=bounds.left()-p.tray_margin
                        if bounds.right()>x1: x1=bounds.right()+p.tray_margin
                        if -bounds.bottom()<y0: y0=-bounds.bottom()-p.tray_margin
                        if -bounds.top()>y1: y1=-bounds.top()+p.tray_margin
                outer=box(x0,y0,x1,y1)
                self.flat.polygon(outer,'#45637b','#e1e8ed').setZValue(-10)
                preview_out_of_bounds=x1-x0>p.print_width or y1-y0>p.print_height
                message=f'备料盘 {x1-x0:.1f} × {y1-y0:.1f} mm · {len(pockets)} 个槽'
                if conflicts: message+=f' · {len(conflicts)} 对凹槽相交或隔墙不足；相交槽将连通'
                if preview_out_of_bounds: message+=' · 标注可能超出打印范围，生成时精确检查'
            else:
                outer=outline.buffer(p.margin,quad_segs=30); self.flat.polygon(outer,'#557b89','#e1e8ed')
                ring=outline.buffer(p.pcb_gap+p.wall,quad_segs=30).difference(outline.buffer(p.pcb_gap,quad_segs=30))
                for gap in gap_polygons(board,self.side(),p): ring=ring.difference(gap)
                self.flat.polygon(ring,'#168994','#8bcfc7'); self.flat.polygon(outline,'#658a9e')
                x0,y0,x1,y1=outer.bounds; message=f'夹具 {x1-x0:.1f} × {y1-y0:.1f} mm · 总高 {p.base+p.wall_height:.2f} mm'
                if x1-x0>p.print_width or y1-y0>p.print_height: issues.append('超出配置的打印范围')
                if p.wall_height>p.pcb_thickness+p.stencil_thickness+.001: message+=' · 侧壁高于钢网，请检查刮刀避让'
            if getattr(self,'flat_needs_fit',True): self.flat.fit(); self.flat_needs_fit=False
            if self.current_tool=='tray': self.sync_position_cells(p)
            self.status.setText(message+('\n'+'；'.join(issues[:4])+(f'……共 {len(issues)} 项' if len(issues)>4 else '') if issues else '\n可生成实体。配合尺寸仍需实际打印确认。'))
            unknown=self.current_tool=='tray' and any(c.mounting=='unknown' and c.side==self.side() for c in board.components)
            self.generate.setEnabled(available[self.side()] and not issues and not unknown)
        except Exception as exc:
            self.generate.setEnabled(False); self.status.setText(str(exc))
    def generate_model(self):
        if self.current_tool=='inventory':
            self.refresh(); return
        try: p=self.params(); board=self.board(); side=self.side(); tool=self.current_tool
        except Exception as exc: self.error(str(exc)); return
        def work(progress):
            solid,warnings=build(board,side,tool,p,progress)
            progress(86,'生成三维预览网格…')
            vertices,faces=solid.tessellate(.20,.35)
            progress(95,'准备三维显示数据…')
            mesh=([v.toTuple() for v in vertices],faces)
            progress(98,'装载三维视图…')
            return solid,warnings,mesh,p
        def done(result):
            self.solid,self.warnings,mesh,self.generated_parameters=result
            try:
                self.tabs.setTabEnabled(1,True); self.tabs.setTabText(1,'三维实体')
                self.tabs.setCurrentIndex(1); self.three.set_mesh(mesh)
            except Exception:
                self.changed()
                raise
            self.export.setEnabled(True)
            self.status.setText('实体已生成并通过有效性检查。'+(' '+ '；'.join(self.warnings[:3]) if self.warnings else ''))
        self.run_job(work,done,'生成实体并预览')
    def export_model(self):
        if self.current_tool=='inventory':
            board=self.board()
            if not board: return
            suggested=DATA_DIR/(board.title+'_找料清单.png')
            path,_=QFileDialog.getSaveFileName(self,'导出找料清单图片',str(suggested),'PNG 图片 (*.png)')
            if not path: return
            if not Path(path).suffix: path+='.png'
            try:
                save_inventory_png(path,board.title,self.inventory_rows)
                self.status.setText('清单图片已保存：'+str(path))
            except Exception as exc: self.error(str(exc))
            return
        if self.solid is None: return
        directory=QFileDialog.getExistingDirectory(self,'选择导出目录',str(DATA_DIR))
        if not directory: return
        self.run_job(lambda progress:export_pair(self.solid,directory,self.board(),self.side(),self.current_tool,self.generated_parameters,self.source,self.source_hash,self.warnings,progress),
                     lambda path:self.status.setText('导出成功（含 STEP 回读验证）：'+str(path)),'导出 STL + STP')
    def reset_parameters(self):
        self.apply_parameters(asdict(Parameters())); self.parameter_changed()
    def apply_parameters(self,values):
        self.loading=True
        for key,control in self.controls.items(): control.setValue(values.get(key,getattr(Parameters(),key)))
        self.labels.setChecked(values.get('labels',True)); self.set_gaps(values.get('gaps',Parameters().gaps)); self.loading=False
    def update_config_title(self):
        name=self.current_config_path.name if self.current_config_path else '未命名'
        suffix=' *' if self.config_dirty else ''
        self.setWindowTitle(f'PCB 贴装助手 — {name}{suffix}')
        if hasattr(self,'config_label'):
            self.config_label.setText(f'配置：{name}{suffix}')
            self.config_label.setToolTip(str(self.current_config_path) if self.current_config_path else '尚未保存配置文件')
    def remember_config(self,path):
        RECENT_CONFIG.parent.mkdir(parents=True,exist_ok=True)
        reference=None
        if path:
            resolved=Path(path).resolve()
            try: reference=resolved.relative_to(APP_DIR.resolve()).as_posix()
            except ValueError: reference=str(resolved)
        RECENT_CONFIG.write_text(json.dumps({'last_config':reference},ensure_ascii=False,indent=2),encoding='utf-8')
    def read_config(self,path):
        state=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(state,dict) or state.get('schema') not in (1,2):
            raise ValueError('不支持的配置版本。')
        values=state.get('parameters')
        if not isinstance(values,dict): raise ValueError('配置中缺少参数。')
        defaults=asdict(Parameters()); defaults={key:defaults[key] for key in CONFIG_KEYS}; result=defaults.copy()
        for key,default in defaults.items():
            if key not in values: continue
            value=values[key]
            if key=='gaps':
                if not isinstance(value,list) or not value or any(
                    not isinstance(row,(list,tuple)) or len(row)!=2 or
                    any(not isinstance(number,(int,float)) or isinstance(number,bool) or not math.isfinite(number)
                        for number in row) for row in value):
                    raise ValueError('配置中的缺口数据无效。')
                if any(not 0<=row[0]<100 or row[1]<=0 for row in value):
                    raise ValueError('配置中的缺口位置或宽度无效。')
            elif isinstance(default,bool):
                if not isinstance(value,bool): raise ValueError(f'配置中的 {key} 无效。')
            elif not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
                raise ValueError(f'配置中的 {key} 无效。')
            elif not 0<=value<=10000 or (key=='spacing' and not 1<=value<=30):
                raise ValueError(f'配置中的 {key} 超出可编辑范围。')
            result[key]=value
        return result
    def save_config(self,path):
        path=Path(path).resolve()
        values=self.config_values()
        data={'schema':2,'parameters':values}
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        try:
            temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)
        self.current_config_path=path; self.saved_parameters=values; self.config_dirty=False
        self.update_config_title(); self.remember_config(path)
        self.status.setText('配置已保存：'+str(path))
        return True
    def save_dialog(self):
        if self.current_config_path is None: return self.save_as_dialog()
        try: return self.save_config(self.current_config_path)
        except Exception as exc: self.error(str(exc)); return False
    def save_as_dialog(self):
        suggested=self.current_config_path or DATA_DIR/'配置.json'
        path,_=QFileDialog.getSaveFileName(self,'配置另存为',str(suggested),'配置 (*.json)')
        if not path: return False
        if not Path(path).suffix: path+='.json'
        try: return self.save_config(path)
        except Exception as exc: self.error(str(exc)); return False
    def confirm_config_change(self):
        if not self.config_dirty: return True
        choice=self.ask_unsaved_config()
        if choice=='save': return self.save_dialog()
        return choice=='discard'
    def ask_unsaved_config(self):
        dialog=QMessageBox(self); dialog.setWindowTitle('未保存的配置')
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText('左侧参数已修改，是否先保存配置？')
        save=dialog.addButton('保存',QMessageBox.ButtonRole.AcceptRole)
        discard=dialog.addButton('放弃修改',QMessageBox.ButtonRole.DestructiveRole)
        cancel=dialog.addButton('取消',QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(save); dialog.setEscapeButton(cancel)
        dialog.exec()
        return 'save' if dialog.clickedButton() is save else ('discard' if dialog.clickedButton() is discard else 'cancel')
    def restore_dialog(self):
        directory=str(self.current_config_path.parent if self.current_config_path else DATA_DIR)
        path,_=QFileDialog.getOpenFileName(self,'打开配置',directory,'配置 (*.json)')
        if path: self.open_config(path)
    def open_config(self,path,ask_to_save=True,quiet=False):
        try:
            path=Path(path).resolve(); values=self.read_config(path)
            if ask_to_save and not self.confirm_config_change(): return False
            self.apply_parameters(values)
            self.current_config_path=path; self.saved_parameters=self.config_values(); self.config_dirty=False
            self.changed(); self.update_config_title(); self.remember_config(path)
            self.status.setText('配置已打开：'+str(path))
            return True
        except Exception as exc:
            if quiet: self.status.setText('上次配置未能打开：'+str(exc))
            else: self.error(str(exc))
            return False
    def restore_recent_config(self):
        if RECENT_CONFIG.exists():
            try:
                state=json.loads(RECENT_CONFIG.read_text(encoding='utf-8'))
                path=state.get('last_config')
                if path:
                    config_path=Path(path)
                    if not config_path.is_absolute(): config_path=APP_DIR/config_path
                    self.open_config(config_path,ask_to_save=False,quiet=True)
            except Exception as exc: self.status.setText('上次配置记录无效：'+str(exc))
            return
        if SETTINGS.exists():
            try:
                values=self.read_config(SETTINGS)
                self.apply_parameters(values); self.config_dirty=self.config_values()!=self.saved_parameters
                self.changed(); self.update_config_title()
                self.status.setText('已读取旧会话中的参数；如需保留，请另存为配置。')
                self.remember_config(None)
            except Exception as exc: self.status.setText('旧会话参数未能读取：'+str(exc))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning(): event.ignore(); return
        if self.confirm_config_change(): event.accept()
        else: event.ignore()


def main():
    app=QApplication.instance() or QApplication([]); configure_fonts(app)
    app.setStyle('Fusion'); configure_appearance(app); window=MainWindow(); window.show(); return app.exec()
