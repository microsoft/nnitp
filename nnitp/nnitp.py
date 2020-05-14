#
# Copyright (c) Microsoft Corporation.
#

import numpy as np
from threading import Thread
from traits.api import HasTraits,String,Enum,Instance,Int,Button
import traits.api as t
from traitsui.api import View, Item, SetEditor, Group, Tabbed, Handler
import model_mgr
from matplotlib.figure import Figure
from mpl_editor import MPLFigureEditor
from itp import LayerPredicate, is_max, Not
from typing import List,Optional,Tuple
from img import prepare_images
import bayesitp
from copy import copy
from PyQt5.QtWidgets import QMenu
from PyQt5.QtGui import QTextCursor
#
# Computation threads. We do computations in threads to avoid freezing
# the GUI.
#

# This thread is for loading the model.

class LoadThread(Thread):
    top : 'MainWindow'
    name : str
    data_model : model_mgr.DataModel
    
    def run(self):
        self.top.message('Loading model "{}"...\n'.format(self.name))
        self.data_model.load(self.name)
        self.top.message('Done loading model.\n')
        self.top.model_loaded()

    def __init__(self,top:'MainWindow',name:str,data_model:model_mgr.DataModel):
        super(LoadThread, self).__init__()
        self.top,self.name,self.data_model = top,name,data_model

# Object specifying an interpolant computation

class InterpolantSpec(object):
    state : 'NormalState'  # new state after interpolant computed
    layer : int
    train_eval : bayesitp.ModelEval
    test_eval : bayesitp.ModelEval
    kwargs : dict

# This thread is for computing an interpolant. 

class InterpolantThread(Thread):
    top : 'MainWindow'
    spec : InterpolantSpec
    data_model : model_mgr.DataModel
    
    def run(self):
        self.top.message('Computing interpolant"{}"...\n'.format(self.name))
        with self.data_model.session():
            spec = self.spec
            itp,stats = bayesitp.interpolant(spec.train_eval,spec.test_eval,spec.layer,
                                             spec.state.input,spec.state.conc,**spec.kwargs)
            self.top.message('Interpolant: {}\n'.format(itp))
            self.top.message(str(stats))
            self.spec.state.itp = itp
            self.spec.state.stats = stats
            self.top.push_state(self.spec.state)

    def __init__(self,top:'MainWindow',spec:InterpolantSpec,data_model:model_mgr.DataModel):
        super(InterpolantThread, self).__init__()
        self.top,self.spec,self.data_model = top,spec,data_model

#        
# GUI elements
#

# A text display just show a text string in a box. This has the
# property that, whenever the text is changed, the cursor is moved to
# the end, so that the last text is visible. This is used to show log
# mesages to the user.

class TextDisplayHandler(Handler):

    def object_string_changed(self,info):
        print ('got here" {}'.format(type(info.string.control)))
        info.string.control.moveCursor(QTextCursor.End)        

    # def setattr(self, info, object, name, value):
    #     Handler.setattr(self, info, object, name, value)
    #     print ('got here')
    #     info.object._updated = True

    # def object__updated_changed(self, info):
    #     if info.initialized:
    #         info.ui.title += "*"

class TextDisplay(HasTraits):
    string =  String('Choose a model name.\n')
    view= View( Item('string',show_label=False, height=25, springy=False,
                     style='custom'), handler = TextDisplayHandler())

# This is the model interface. It has a combo box for selecting the
# model from the available models. When the user selects a model name,
# the load thread is fired up to load it. Initially, the model name is
# 'none`.

class Model(HasTraits):
    name = Enum('none',*list(model_mgr.datasets),desc='Available models and datasets')
    top : 'MainWindow'
    data_model = Instance(model_mgr.DataModel)
    worker_thread = Instance(Thread)
    use_set = Enum('training','test',desc='Dataset to view')
    size = Int(100)
    _tf_session = None
    _train_eval : bayesitp.ModelEval
    _test_eval : bayesitp.ModelEval
    
    def _data_model_default(self):
        return model_mgr.DataModel()

    view = View(Item('name',style='simple'),Item('use_set',style='simple'),'size')

    def check_busy(self) -> bool:
        if self.worker_thread and self.worker_thread.isAlive():
            self.top.message('The model is busy.')
            return False
        return True

    def _name_changed(self):
        if self.check_busy():
            self.worker_thread = LoadThread(self.top,self.name,self.data_model)
            self.worker_thread.start()

    def _size_changed(self):
        if self.data_model.loaded:
            self._train_eval = bayesitp.ModelEval(self.data_model.model,
                                                  self.data_model.x_train[:self.size])
            self._test_eval = bayesitp.ModelEval(self.data_model.model,
                                                 self.data_model.x_test[:self.size])

    def _use_set_changed(self):
        self.top.update()

    def inputs(self):
        return (self.data_model.x_train if self.use_set == 'training'
                else self.data_model.x_test)
        
    def model_eval(self):
        return (self._train_eval if self.use_set == 'training' else self._test_eval)

    def layers(self):
        if not self.data_model.loaded:
            return []
        return ['-1:input'] + [str(i)+':'+str(type(l))
                               for i,l in enumerate(self.data_model.model.layers)]

    def output_layer(self) -> int:
        return len(self.data_model.model.layers) - 1
        
    def get_inputs_pred(self,pred:LayerPredicate,N=9):
        eval = self.model_eval()
        eval.set_pred(pred.layer,pred.pred)
        return list(eval.indices()[:N]),list(eval.split(-1)[0][:N])

    # Get the index of the previous layer in the layers list. If no
    # previous layer, returns -1, for the input layer.

    def previous_layer(self,layer:int) -> int:
        layers = [int(x.split(':')[0]) for x in self.top.layers]
        layers = [x for x in layers if x < layer]
        return layers[-1] if layers else -1

    # Compute an interpolant for a given input and conclusion in a
    # thread, advancing the state when completed. Here `state` is the
    # new state into which the interpolant should be inserted.

    def interpolant(self,state):
        if self.check_busy():
            spec = InterpolantSpec()
            spec.state = state
            spec.layer =  self.previous_layer(state.conc.layer)
            spec.train_eval = self._train_eval
            spec.test_eval = self._test_eval
            spec.kwargs = dict()
            self.worker_thread = InterpolantThread(self.top,spec,self.data_model)
            self.worker_thread.start()

    def pred_cone(self,lp:LayerPredicate) -> Tuple:
        return bayesitp.get_pred_cone(self.data_model.model,lp)

            
# Metadata for images displayed in the figure.

class FigMetaData(object):
    input_idx: int
    input : np.ndarray
    pred : LayerPredicate
    cone : Optional[Tuple] = None

    def __init__(self,input_idx,input,pred,cone=None):
        self.input_idx,self.input,self.pred,self.cone = (input_idx,input,pred,cone)
    
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
    
    # Draw the matplotlib figure corresponding to this state

    def render(self,figure:Figure):

        # In the initial state, we display a choice of input images
        # satisfying the conclusion and allow the user to select one.

        rows,cols = 5,10
        imgs = prepare_images(self.compset[:(rows*cols)])
        for idx,img in enumerate(imgs):
            if idx >= rows*cols:
                break
            sub = figure.add_subplot(rows, cols, idx + 1)
            sub.imshow(img)
            sub.axis('off')
            sub.identifier = FigMetaData(idx,img,self.conc)
        figure.canvas.draw()

    # When the user clicks on an image in the figure, we set that
    # image as our input and transition to a normal state.

    def onclick(self,event):
        if event.inaxes is not None:
            id = event.inaxes.identifier.input_idx
            if id < len(self.compset):
                self.top.message('Selecting input image {}. Click the image to explain.\n'.format(id))
                new_state = NormalState()
                new_state.top = self.top
                new_state.input_idx = id
                new_state.input = self.compset[id]
                new_state.conc = self.conc
                self.top.push_state(new_state)
                
    def on_axes_enter(self,event):
        pass
    
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
    conc : Optional[LayerPredicate] = None
    itp : Optional[LayerPredicate] = None
    comp : Optional[LayerPredicate] = None
    compset : List[int] = []
    compimgs : List[np.ndarray] = []
    stats : bayesitp.Stats
    
    # Draw the matplotlib figure corresponding to this state.  In the
    # normal case, we display the input image along with the
    # comparison set.  If there is an interpolant, then below each
    # image, we display the cone of each conjunct of the interpolant.
    # All the images in the figure are normalozed to the same
    # intensity scale.

    def render(self,figure:Figure):

        if self.comp is None:
            self.set_comparison(self.itp)
        cols = 1 + len(self.compset)
        top_row = [self.input] + self.compimgs
        top_idxs = [self.input_idx] + self.compset
        imgs = list(top_row)
        concs = [self.conc]
        cones = []
        with self.top.model.data_model.session():
            if self.itp is not None:
                conjs = self.itp.conjs()
                for conj in conjs:
                    cone = self.top.model.pred_cone(conj)
                    imgs.extend(im[cone] for im in top_row)
                    cones.append(cone)
                concs.extend(conjs)
        imgs = prepare_images(imgs) 
        rows = len(imgs) // cols
        for idx,img in enumerate(imgs): 
            row,col = idx//cols, idx % cols
            sub = figure.add_subplot(rows, cols, idx + 1)
            sub.imshow(img)
            sub.axis('off')
            sub.identifier = FigMetaData(top_idxs[col],top_row[col],concs[row])
            if row == 0:
                for sidx,slc in enumerate(cones):
                    ycenter = (slc[0].start + slc[0].stop)/2.0
                    xcenter = (slc[1].start + slc[1].stop)/2.0
                    pixel = img[int(ycenter),int(xcenter)]
                    c = 'black' if np.mean(pixel) >= 0.5 else 'white'
                    sub.text(xcenter,ycenter,str(sidx),
                             verticalalignment='center', horizontalalignment='center', color=c)
        figure.canvas.draw()

    # When the user selects an image we compute an interpolant for the
    # predicate that the image satisfies. If the user selects a cone
    # of an interpolant conjunct, we compute an interpolant for that
    # conjunct. After the interpolant is computed, the resulting state
    # will be pushed.

    def onclick(self,event):
        if event.inaxes is not None:
            id = event.inaxes.identifier
            print ('input_idx = {}, pred = {}'.format(id.input_idx,id.pred))
            new_state = copy(self)
            new_state.comp = None
            new_state.compset = []
            new_state.compimgs = []
            if event.button == 3:
                print ('x = {}, y = {}'.format(event.x,event.y))
                canvas = self.top.figure.canvas
                menu = QMenu(canvas)
                examplesAction = menu.addAction("Examples")
                counterexamplesAction = menu.addAction("Counterexamples")
                thing = canvas.mapToGlobal(event.guiEvent.pos())
                action = menu.exec_(thing)
                if action == examplesAction:
                    new_state.set_comparison(id.pred)
                elif action == counterexamplesAction:
                    new_state.set_comparison(id.pred.negate())
                else:
                    return
                self.top.message('Comparison predicate: {}\n'.format(id.pred))
                self.top.push_state(new_state)
                return
            new_state.input_idx = id.input_idx
            new_state.conc = id.pred
            new_state.input = id.input
            self.top.model.interpolant(new_state)
                
    def on_axes_enter(self,event):
        if event.inaxes is not None:
            id = event.inaxes.identifier
            self.top.predicate = str(id.pred)

    # Set the comparison predicate and update the comparison list
    # accordingly. This displays the first 9 images satisfying the
    # compariosn predicate, along with their cones.

    def set_comparison(self,comp:Optional[LayerPredicate]):
        self.comp = comp
        if comp is not None:
            self.compset, self.compimgs = self.top.model.get_inputs_pred(comp)
        else:
            self.compset, self.compimgs = [],[]
            
        

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
    figure = Instance(Figure,())
    layers = t.List(editor=SetEditor(name='avail'))
    category = Int(0)
    predicate = String()
    back = Button()

    view = View(
        Item('display',show_label=False, style='custom'),Tabbed(
            Group('model', 'layers', label = 'Model'),
            Group('back', Item('predicate',style='readonly'),
                  Item('figure', editor=MPLFigureEditor(),
                       show_label=False, springy=True), label='Images'),
            style='custom'), resizable=True)
    
    # List of availaible layers, used by `layers`, above
    avail = t.List([])

    # Stack of states
    _states : List[State] = []

    def _model_default(self):
        return Model(top=self)
    
    def __init__(self):
        super(MainWindow, self).__init__()
        
    def update(self):
        self.figure.clear()
        if len(self._states):
            self._states[-1].render(self.figure)
            self.figure.canvas.mpl_connect('button_press_event', self.onclick)
            self.figure.canvas.mpl_connect('axes_enter_event', self.on_axes_enter)
                
    def model_loaded(self):
        state = InitState()
        self.model._size_changed()   # update model evaluators
        state.conc = LayerPredicate(self.model.output_layer(),is_max(self.category))

        # Get the inputs of selected category according to the model.
        self.model.model_eval().set_pred(state.conc.layer,state.conc.pred)
        state.compset,_ = self.model.model_eval().split(-1)

        state.top = self
        self.avail = self.model.layers()
        self.message('Select layers for explanations, then choose an input image in the images tab.\n')
        self.push_state(state)

    def push_state(self,state:State):
        self._states.append(state)
        self.update()

    def message(self,msg:str):
        self.display.string = self.display.string + msg
#        print (self.trait_views())
#        print ('dict = {}'.format(self.display.string.__dict__))
#        self._get_editor('display').control.moveCursor(QTextCursor.End)

    def onclick(self,event):
        print ('onclick: {}'.format(event.button))
        if self._states:
            self._states[-1].onclick(event)
        
    def on_axes_enter(self,event):
        if self._states:
            self._states[-1].on_axes_enter(event)

    def _back_fired(self):
        if len(self._states) > 1:
            self._states.pop()
            self.update()
            
# Display the main window
        
if __name__ == '__main__':
    MainWindow().configure_traits()
