from PyQt5 import QtWidgets, QtGui, QtCore


def toolbar_widget():
    toolbar = QtWidgets.QToolBar()
    toolbar.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    toolbar.setStyleSheet('QToolBar {background-color: #2E2E2E; color: white;}')
    return toolbar


def dock_widget(name=''):
    dock = QtWidgets.QDockWidget(name)
    dock.setStyleSheet('''
        QDockWidget {
            background-color: #222222;
            font-weight: bold;
            font-size: 12px;
            color: #CCCCCC;
        }
        QDockWidget::title {
            text-align: center;
            background-color: #444444;
            padding: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QDockWidget::close-button {
            background-color: #666666;
            icon-size: 12px;
        }
        QDockWidget::close-button:hover {
            background-color: #ff5555;
        }
    ''')
    dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
    dock.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    return dock


def group_widget(name=''):
    group = QtWidgets.QGroupBox(name)
    group.setStyleSheet('''
        QGroupBox {
            background-color: #444444;
            border: 0px solid #444444;
            border-bottom-left-radius: 4px;
            border-bottom-right-radius: 4px;
            margin-top: 0ex;
            margin-bottom: 0ex;
            font-weight: bold;
            font-size: 12px;
            color: #CCCCCC;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 0px;
        }
    ''')
    group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    return group


def create_dock(name=''):
    dock = dock_widget(name)
    group = group_widget()
    dock.setWidget(group)
    return dock, group


def create_file_dialogue(name="Save File", file_filter="All Files (*)", default_dir=""):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    dialog = QtWidgets.QFileDialog()
    dialog.setOptions(options)
    if "Save File" == name:
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    if "Open File" == name:
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dialog.setNameFilters([file_filter])
    dialog.setWindowTitle(name)
    dialog.setDirectory(default_dir)
    dialog.setStyleSheet("""
            QFileDialog {
                background-color: #333333;
                color: white;
            }
            QFileDialog QLabel {
                color: white;
            }
            QFileDialog QLineEdit {
                background-color: #2E2E2E;
                color: white;
                selection-background-color: #0096FF;
            }
            QFileDialog QPushButton {
                background-color: #2E2E2E;
                color: white;
                padding: 5px;
                min-width: 80px;
            }
            QFileDialog QPushButton:hover {
                background-color: #0096FF;
            }
        """)
    return dialog


def create_scroll_area():
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    scroll_area.setStyleSheet("""
                                QScrollArea {
                                    background-color: #444444;
                                    color: white;
                                }
                            """)
    content_widget = QtWidgets.QWidget(scroll_area)
    content_widget.setStyleSheet("background-color: #444444;")
    scroll_area.setWidget(content_widget)
    layout = QtWidgets.QFormLayout(content_widget)
    content_widget.setLayout(layout)
    return scroll_area, layout


def frame_widget(h=True):
    line = QtWidgets.QFrame()
    if h:
        line.setFrameShape(QtWidgets.QFrame.HLine)
    else:
        line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def label_widget(name=''):
    label = QtWidgets.QLabel(name)
    label.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    label.setStyleSheet('background-color: #444444; color: #ECECEC; padding: 2px; border-radius: 2px;')
    return label


def lcdnumber_widget(num=None, n=None):
    lcd = QtWidgets.QLCDNumber()
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))  # set text color
    palette.setColor(QtGui.QPalette.Background, QtGui.QColor(37, 37, 38))  # set background color
    palette.setColor(QtGui.QPalette.Light, QtGui.QColor(64, 64, 64))  # set light color
    palette.setColor(QtGui.QPalette.Dark, QtGui.QColor(26, 26, 27))  # set dark color
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 85, 255))  # set highlight color
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))  # set highlighted text color
    lcd.setPalette(palette)
    lcd.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    lcd.setDecMode()
    if num is not None:
        lcd.display(num)
    if n is not None:
        lcd.setDigitCount(n)
    lcd.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    lcd.setMinimumWidth(lcd.sizeHint().width())
    lcd.setMaximumWidth(64)
    return lcd


def spinbox_widget(range_min, range_max, step, value):
    spinbox = QtWidgets.QSpinBox()
    spinbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    spinbox.setPalette(dark_palette)
    spinbox.setRange(range_min, range_max)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    spinbox.setMinimumWidth(spinbox.sizeHint().width())
    spinbox.setMinimumHeight(spinbox.sizeHint().height())
    max_value_width = spinbox.fontMetrics().width(str(spinbox.maximum()))
    max_value_height = spinbox.fontMetrics().height()
    spinbox.setMaximumWidth(max_value_width + 32)
    spinbox.setMaximumHeight(max_value_height + 16)
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    doublespinbox.setPalette(dark_palette)
    doublespinbox.setRange(range_min, range_max)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    doublespinbox.setValue(value)
    doublespinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    doublespinbox.setMinimumWidth(doublespinbox.sizeHint().width())
    doublespinbox.setMinimumHeight(doublespinbox.sizeHint().height())
    max_value_width = doublespinbox.fontMetrics().width(str(doublespinbox.maximum()))
    max_value_height = doublespinbox.fontMetrics().height()
    doublespinbox.setMaximumWidth(max_value_width + 32)
    doublespinbox.setMaximumHeight(max_value_height + 16)
    return doublespinbox


def pushbutton_widget(name='', checkable=False, enable=True, checked=False):
    button = QtWidgets.QPushButton(name)
    button.setCheckable(checkable)
    button.setEnabled(enable)
    button.setChecked(checked)
    button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    button.setStyleSheet('''
        QPushButton {
            background-color: #282828;
            border-style: outset;
            border-radius: 4px;
            color: #FFFFFF;
            padding: 2px;
        }
        QPushButton:hover {
            background-color: #4169e1;
        }
        QPushButton:pressed {
            background-color: #045c64;
            border-style: inset;
        }
        QPushButton:checked {
            background-color: #a52a2a;
            border-style: inset;
        }
    ''')
    button.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    button_size = button.fontMetrics().boundingRect(button.text()).size()
    if button_size.width() < 96:
        button_size.setWidth(96)
    else:
        button_size.setWidth(button_size.width() + 24)
    if button_size.height() < 24:
        button_size.setHeight(24)
    else:
        button_size.setHeight(button_size.height() + 16)
    button.setFixedSize(button_size.width(), button_size.height())
    return button


def checkbox_widget(name=''):
    checkbox = QtWidgets.QCheckBox(name)
    checkbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    checkbox.setStyleSheet('''
        QCheckBox {
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #fff, stop: 0.9 #ddd);
            color: #000;
            padding: 2px;
            border: 2px solid #aaa;
            border-radius: 20px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border-radius: 20px;
            border: 2px solid #aaa;
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #fff, stop: 0.9 #ddd);
        }
        QCheckBox::indicator:checked {
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #9af, stop: 0.9 #8bf);
            border: 2px solid #8bf;
        }
    ''')
    checkbox.setChecked(False)
    checkbox.setCheckable(True)
    return checkbox


def radiobutton_widget(name='', color=f"rgb(192, 255, 62)", autoex=False):
    radiobutton = QtWidgets.QRadioButton(name)
    radiobutton.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    radiobutton.setAutoExclusive(autoex)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(200, 200, 200))  # set text color
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))  # set background color
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))  # set button color
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197))  # set highlight color
    radiobutton.setPalette(palette)
    radiobutton.setStyleSheet('''
                    QRadioButton {
                        background-color: #444444;
                        color: white;
                    }
                    QRadioButton::indicator {
                        width: 8px;
                        height: 8px;
                    }
                    QRadioButton::indicator::unchecked {
                        border: 2px solid rgb(200, 200, 200);
                        border-radius: 4px;
                    }
                    QRadioButton::indicator::checked {
                        background-color: %s;
                        border: 2px solid %s;
                        border-radius: 4px;
                    }''' % (color, color))
    return radiobutton


def combobox_widget(list_items):
    combobox = QtWidgets.QComboBox()
    for item in list_items:
        combobox.addItem(item)
    combobox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    combobox.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    combobox.setStyleSheet('QComboBox {background-color: #454545; color: #f0f0f0;}'
                           'QComboBox::down-arrow { image: none; }'
                           'QComboBox::drop-down { border: none; }')
    combobox.setMaximumWidth(100)
    return combobox


def lineedit_widget():
    lineedit = QtWidgets.QLineEdit()
    lineedit.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    lineedit.setStyleSheet('''
                    QLineEdit {
                        background-color: #444444;
                        color: white;
                        }
                    ''')
    return lineedit


def text_widget():
    text_edit = QtWidgets.QTextEdit()
    text_edit.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    text_edit.setStyleSheet('''
                        QTextEdit {
                            background-color: #444444;
                            color: white;
                            selection-background-color: #0096FF
                            }
                        ''')
    return text_edit


def slider_widget(mi, ma, value, step):
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setMinimum(mi / step)
    slider.setMaximum(ma / step)
    slider.setSingleStep(1)
    slider.setValue(value)
    slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
    slider.setTickInterval(1 / step)
    slider.setStyleSheet("""
                        QSlider::groove:horizontal {
                            height: 8px;
                            background-color: #222222;
                            border: 1px solid #222222;
                            border-radius: 2px;
                        }
                
                        QSlider::handle:horizontal {
                            width: 8px;
                            height: 16px;
                            background-color: #FFFFFF;
                            border: 1px solid #222222;
                            border-radius: 6px;
                            margin: -5px 0;
                        }
                
                        QSlider::sub-page:horizontal {
                            height: 8px;
                            background-color: #222222;
                            border: 1px solid #222222;
                            border-radius: 2px;
                        }
                        """)
    return slider


def dia_widget(mi, ma, value):
    dial = QtWidgets.QDial()
    dial.setMinimum(mi)
    dial.setMaximum(ma)
    dial.setValue(value)
    dial.setStyleSheet('''
        QDial {
            background-color: #303030;
            border: none;
        }
        QDial::handle {
            background-color: #777777;
            border-radius: 6px;
        }
        QDial::handle:hover {
            background-color: #999999;
        }
        QDial::handle:pressed {
            background-color: #555555;
        }
    ''')
    return dial


def dialog(labtex=False):
    dialogue = QtWidgets.QDialog()
    dialogue.setStyleSheet(''' 
        QDialog {
            background-color: #333;
            color: #FFF;
        }
        ''')
    dialogue.setWindowTitle("Please Wait")
    layout = QtWidgets.QVBoxLayout()
    label = QtWidgets.QLabel("Task is running, please wait...")
    layout.addWidget(label)
    dialogue.setLayout(layout)
    dialogue.setModal(True)
    if labtex:
        return dialogue, label
    else:
        return dialogue


def message_box(title):
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle(title)
    msg.setStandardButtons(QtWidgets.QMessageBox.NoButton)
    msg.setStyleSheet("""
        QMessageBox {
            background-color: #333;
            color: #EEE;
            text-align: center;
        }
    """)
