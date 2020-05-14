#
# Copyright (c) Microsoft Corporation.
#

from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt4agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from traits.api import Any, Instance
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        return FigureCanvas(self.value)

class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


