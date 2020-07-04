#
# Copyright (c) Microsoft Corporation.
#

import numpy as np
from threading import Thread
from traits.api import HasTraits,String,Enum,Instance,Int,Button,Float,Bool
import traits.api as t
from traitsui.api import View, Item, SetEditor, Group, Tabbed, Handler
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from .model_mgr import DataModel,datasets,import_models
from .itp import LayerPredicate, AndLayerPredicate, BoundPredicate
from .itp import output_category_predicate
from typing import List,Optional,Tuple
from .bayesitp import Stats, interpolant, get_pred_cone, fraction, fractile
from copy import copy
from PyQt5.QtCore import Qt,Signal,QObject
from PyQt5.QtWidgets import QMenu,QApplication,QFrame,QHBoxLayout,QWidget
from PyQt5.QtGui import QTextCursor
import sys

#
# Computation threads. We do computations in threads to avoid freezing
# the GUI.
#

# This thread is for loading the model.

class LoadThread(Thread):
    top : 'MainWindow'
    name : str
    data_model : DataModel
    
    def run(self):
        self.top.message('Loading model "{}"...\n'.format(self.name))
        self.data_model.load(self.name)
        self.top.message('Done loading model.\n')
        self.top.signals.model_loaded.emit()

    def __init__(self,top:'MainWindow',name:str,data_model:DataModel):
        super(LoadThread, self).__init__()
        self.top,self.name,self.data_model = top,name,data_model

# Object specifying an interpolant computation

class InterpolantSpec(object):
    state : 'NormalState'  # new state after interpolant computed
    layer : int
    data_model : DataModel
    kwargs : dict

# This thread is for computing an interpolant. 

class InterpolantThread(Thread):
    top : 'MainWindow'
    spec : InterpolantSpec
    
    def run(self):
        self.top.message('Computing interpolant...\n')
        with self.spec.data_model.session():
            spec = self.spec
            input = spec.state.input.reshape(1,*spec.state.input.shape)
            itp,stats = interpolant(self.spec.data_model,spec.layer,
                                             input,spec.state.conc,**spec.kwargs)
            self.top.message('Interpolant: {}\n'.format(itp))
            self.top.message(str(stats))
            self.spec.state.itp = itp
            self.spec.state.stats = stats
            self.top.signals.push_state.emit(self.spec.state)

    def __init__(self,top:'MainWindow',spec:InterpolantSpec):
        super(InterpolantThread, self).__init__()
        self.top,self.spec = top,spec

#        
# GUI elements
#

# A text display just show a text string in a box. This has the
# property that, whenever the text is changed, the cursor is moved to
# the end, so that the last text is visible. This is used to show log
# mesages to the user.

class TextDisplayHandler(Handler):

    def object_string_changed(self,info):
        info.string.control.moveCursor(QTextCursor.End)        


class TextDisplay(HasTraits):
    string =  String('Choose a model name.\n')
    view= View( Item('string',show_label=False, height=25, springy=False,
                     style='custom'), handler = TextDisplayHandler())

# This is the model interface. It has a combo box for selecting the
# model from the available models. When the user selects a model name,
# the load thread is fired up to load it. Initially, the model name is
# 'none`.

class Model(HasTraits):

    top : 'MainWindow'  # reference to the main window

    # Combo box for model name

    name = Enum('none',*list(datasets),desc='Available models and datasets')

    # Choice of dataset to view

    use_set = Enum('training','test',desc='Dataset to view')
    
    # Interpolation parameters
    
    alpha = Float(0.98)  # precision parameter
    gamma = Float(0.6)   # recall parameter
    mu = Float(0.9)      # recall shrink parameter
    size = Int(100)      # max training sample size
    ensemble_size = Int(1) # size of interpolant ensemble

    # Internal state

    data_model : DataModel = Instance(DataModel)
    worker_thread = Instance(Thread)
    _tf_session = None
    _param_names : List[str] = ['alpha','gamma','mu','size','ensemble_size']

    
    def _data_model_default(self):
        return DataModel()

    view = View(Item('name',style='simple'),Item('use_set',style='simple'),'alpha','gamma','mu','size','ensemble_size')

    def check_busy(self) -> bool:
        if self.worker_thread and self.worker_thread.isAlive():
            self.top.message('The model is busy.')
            return False
        return True

    def _name_changed(self):
        if self.check_busy():
            self.worker_thread = LoadThread(self.top,self.name,self.data_model)
            QApplication.setOverrideCursor(Qt.BusyCursor)
            self.worker_thread.start()

    def _size_changed(self):
        if self.data_model.loaded:
            self.data_model.set_sample_size(self.size)

    def _data_model_changed(self):
        self.set_kwargs(self.data_model.params)

    def _use_set_changed(self):
        self.top.update()

    def inputs(self):
        return (self.data_model.x_train if self.use_set == 'training'
                else self.data_model.x_test)
        
    def model_eval(self):
        return (self.data_model._train_eval if self.use_set == 'training'
                else self.data_model._test_eval)

    def layers(self):
        if not self.data_model.loaded:
            return []
        return ['-1:input'] + ['{:0>2}'.format(i)+':'+ l
                               for i,l in enumerate(self.data_model.model.layers)]

    def output_layer(self) -> int:
        return self.data_model.output_layer()
        
    def get_inputs_pred(self,lpred:LayerPredicate,N=9):
        eval = self.model_eval()
        eval.set_layer_pred(lpred)
        return list(eval.indices()[:N]),list(eval.split(-1)[0][:N])

    # Get the index of the previous layer in the layers list. If no
    # previous layer, returns -1, for the input layer.

    def previous_layer(self,layer:int) -> int:
        layers = [int(x.split(':')[0]) for x in self.top.layers]
        layers = [x for x in sorted(layers) if x < layer]
        return layers[-1] if layers else -1

    # Compute an interpolant for a given input and conclusion in a
    # thread, advancing the state when completed. Here `state` is the
    # new state into which the interpolant should be inserted.

    def interpolant(self,state):
        if self.check_busy():
            spec = InterpolantSpec()
            spec.state = state
            spec.layer =  self.previous_layer(state.conc.layer)
            spec.data_model = self.data_model
            kwargs = self.get_kwargs()
            spec.kwargs = dict((k,kwargs[k]) for k in ['alpha','gamma','mu','ensemble_size'])
            self.worker_thread = InterpolantThread(self.top,spec)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.worker_thread.start()

    # Set traits from the interpolation keyword args

    def set_kwargs(self,kwargs):
        for key,val in kwargs.items():
            if key in self._param_names:
                self.__setattr__(key,val)
                

    # Get the interpolant keyword args from object traits

    def get_kwargs(self):
        return dict((key,self.__dict__[key]) for key in self._param_names)

    def pred_cone(self,lp:LayerPredicate) -> Tuple:
        return get_pred_cone(self.data_model.model,lp)

            
# Base class for interface states. A state object has a reference
# `top` to the main UI window.

class State(object):
    top : 'MainWindow'
    
# An InitState consists of the following elements:
#
# - A conclusion (a layer/predicate pair)
# - A comparison set.
#
# The comparison set is the list of input indices satisfying the comparison
# condition.

class InitState(State):
    conc : Optional[LayerPredicate] = None
    compset : List[int] = []

    _displayed_category : Optional[int] = None

    # Draw the grid corresponding to this state

    def render(self):

        # In the initial state, we display a choice of input images
        # satisfying in the chosen category and allow the user to select one.

        # Get the inputs of selected category according to the model.
        category = self.top.category
        if self._displayed_category != category:
            self.conc = output_category_predicate(self.top.model.data_model,category)
            with self.top.model.data_model.session():
                self.compset = self.conc.sat(self.top.model.model_eval())
            self._displayed_category = category

        # Render the comparison set.
        
        data_idxs = [(i,) for i in range(len(self.compset))]
        # TODO: `data_idxs` should be the real indexes in the dataset.
        canvas = self.top.model.data_model.datatype.offer(self.compset,data_idxs,self)
        self.top.figure.add(canvas)

    # When the user clicks on an image in the figure, we set that
    # image as our input and transition to a normal state.

    def onclick(self,id,canvas,event):
        input_idx = id.input_idx[0]
        print ('button = {}'.format(event.button()))
        if input_idx < len(self.compset):
            self.top.message('Selecting input image {}. Click the image to explain.\n'.format(id))
            new_state = NormalState()
            new_state.top = self.top
            new_state.input_idx = input_idx
            new_state.input = self.compset[input_idx]
            new_state.conc = self.conc
            new_state.category_pred = self.conc
            print ('new_state.input.shape = {}'.format(new_state.input.shape))
            self.top.push_state(new_state)
            print (2)
                
    def on_axes_enter(self,id,canvas,event):
        pass
    
    def new_figure(self):
        return self.top.figure.new_figure(self)

# A NormalState consists of the following elements:
#
# - An input valuation (the premise)
# - A conclusion (a layer/predicate pair)
# - An interpolant (a layer/predicate pair)
# - A comparison condition (a layer/predicate pair)
# - A comparison set.
# - Interpolation statsistics
#
# The comparison set is the list of input indices satisfying the comparison
# condition.

class NormalState(State):
    input_idx : int = 0  # index of input in dataset
    input : np.ndarray = None
    conc : LayerPredicate
    category_pred : LayerPredicate
    itp : Optional[LayerPredicate] = None
    comp : Optional[LayerPredicate] = None
    compset : List[int] = []
    compimgs : List[np.ndarray] = []
    stats : Stats
    _restrict : bool = False
    
    # Draw the matplotlib figure corresponding to this state.  In the
    # normal case, we display the input image along with the
    # comparison set.  If there is an interpolant, then below each
    # image, we display the cone of each conjunct of the interpolant.
    # All the images in the figure are normalized to the same
    # intensity scale.

    def render(self):

        if self.comp is None:
            self.set_comparison(self.itp)
        elif self.restrict != self.top.restrict:
            self.refresh_comparison()
        top_row = [self.input] + self.compimgs
        top_idxs = [(x,) for x in ([self.input_idx] + self.compset)]
        canvas = self.top.model.data_model.datatype.render(top_row,top_idxs,self)
        self.top.figure.add(canvas)


    # When the user selects an image we compute an interpolant for the
    # predicate that the image satisfies. If the user selects a cone
    # of an interpolant conjunct, we compute an interpolant for that
    # conjunct. After the interpolant is computed, the resulting state
    # will be pushed.

    def onclick(self,id,canvas,event):
        new_state = copy(self)
        new_state.comp = None
        new_state.compset = []
        new_state.compimgs = []
        if event.button() == 2:
            menu = QMenu(canvas)
            examplesAction = menu.addAction("Examples")
            counterexamplesAction = menu.addAction("Counterexamples")
            thing = canvas.mapToGlobal(event.pos())
            action = menu.exec_(thing)
            if action == examplesAction:
                new_state.set_comparison(id.pred)
            elif action == counterexamplesAction:
                new_state.set_comparison(id.pred.negate())
            else:
                return
            self.top.push_state(new_state)
            return
        new_state.input_idx = id.input_idx[0]
        new_state.conc = id.pred
        new_state.input = id.input
        self.top.model.interpolant(new_state)
                
    def on_axes_enter(self,id,canvas,event):
        self.top.predicate = str(id.pred)
        self.top.fraction = fraction(self.top.model.model_eval(),id.pred)

    # Set the comparison predicate and update the comparison list
    # accordingly. This displays the first 9 images satisfying the
    # compariosn predicate, along with their cones.

    def set_comparison(self,comp:Optional[LayerPredicate]):
        self.comp = comp
        if comp is not None:
            self.refresh_comparison()
        else:
            self.compset, self.compimgs = [],[]
            
    def refresh_comparison(self):
        if self.comp is not None:
            comp = self.comp
            if isinstance(comp.pred,BoundPredicate):
                percentile = self.top.percentile
                if percentile > 0.0 and percentile <= 100.0:
                    comp = fractile(self.top.model.model_eval(),
                                             comp.layer,comp.pred.var,comp.pred.pos,
                                             percentile/100.0)
            if self.top.restrict:
                comp = AndLayerPredicate(comp,self.category_pred)
            self.top.message('Comparison predicate: {}\n'.format(comp))
            self.compset, self.compimgs = self.top.model.get_inputs_pred(comp)
            self.restrict = self.top.restrict

    def new_figure(self):
        return self.top.figure.new_figure(self)


# This "editor" is just a blank frame in which to draw the image grid.

class _ImageGridEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = QFrame()
        self.control.setLayout(QHBoxLayout())
        self.value.frame = self.control
        self.set_tooltip()

    def update_editor(self):
        pass

class ImageGridEditor(BasicEditorFactory):

    klass = _ImageGridEditor

# This is the ImageGrid object. It has a frame and a list of child
# widgets, layed out horizontally.

class ImageGrid(object):
    frame : QFrame
    children : List[QWidget] = [] 

    def clear(self):
        for child in self.children:
            child.setParent(None)
        del self.children[:]

    def add(self,canvas:QWidget):
        self.frame.layout().addWidget(canvas)
        self.children.append(canvas)
        


class Signals(QObject):
    model_loaded = Signal()
    push_state = Signal(State)

# The main window. The view consists of the following elements:
#
# - A `display` for status updates.
# - A `model`
# - A matplotlib `figure` for displaying the top state
# - A list `layers` of layers at which to compute interpolants.
# - An integer category
#
# The data consists of
#
# - A stack of `_states`, intially empty
#

class MainWindow(HasTraits):
    display = Instance(TextDisplay(), ())
    model = Instance(Model)
    figure = Instance(ImageGrid,())
    layers = t.List(editor=SetEditor(name='avail'))
    category = Int(0)
    restrict = Bool()
    predicate = String()
    fraction = Float()
    percentile = Float(0.0)
    back = Button()

    view = View(
        Item('display',show_label=False, style='custom',style_sheet='*{font-size:11pt}'),Tabbed(
            Group('model', 'layers', label = 'Model'),
            Group(Group('category','restrict',Item('back',show_label=False),
                        orientation='horizontal',style='simple'),
                  Group(Item('predicate',style='readonly'),
                        Item('fraction',style='readonly'),
                        Item('percentile',style='simple'),orientation='horizontal'),
                  Item('figure', editor=ImageGridEditor(),
                       show_label=False, springy=True), label='Images'),
            style='custom', style_sheet='*{font-size:11pt}'), resizable=True)
    
    # List of availaible layers, used by `layers`, above
    avail = t.List([])

    # Stack of states
    _states : List[State] = []
    signals : Signals

    def _model_default(self):
        return Model(top=self)
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.signals = Signals()
        self.signals.model_loaded.connect(self.model_loaded)
        self.signals.push_state.connect(self.push_state)
        
    def update(self):
        self.figure.clear()
        if len(self._states):
            self._states[-1].render()
                
    def model_loaded(self):
        state = InitState()
        self.model._data_model_changed()  # update the interpolation parameters
        self.model._size_changed()        # update model evaluators
        state.top = self
        self.avail = self.model.layers()
        layer_idxs = self.model.data_model.params.get('layers',[])
        self.layers = list_elems(self.avail,[i+1 for i in layer_idxs])
        self.message('Select layers for explanations, then choose an input image in the images tab.\n')
        self.push_state(state)
        
    def push_state(self,state:State):
        self._states.append(state)
        self.update()
        QApplication.restoreOverrideCursor()

    def message(self,msg:str):
        self.display.string = self.display.string + msg

    def onclick(self,event):
        if self._states:
            self._states[-1].onclick(event)
        
    def on_axes_enter(self,event):
        if self._states:
            self._states[-1].on_axes_enter(event)

    def _back_fired(self):
        if len(self._states) > 1:
            self._states.pop()
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.update()
            QApplication.restoreOverrideCursor()
            
    def _restrict_changed(self):
        self.update()
        
    def _category_changed(self):
        self.update()

def list_elems(l1,l2):
    return [l1[i] for i in l2 if i >= 0 and i < len(l1)]

def main():
    import_models()
    if len(sys.argv) > 1:
        verb = sys.argv[1]
        if verb == 'compress':
            from .compress import main as compress_main
            return compress_main()
    MainWindow().configure_traits()
        
# Display the main window
        
if __name__ == '__main__':
    main()
