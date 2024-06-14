from PyQt5 import QtWidgets, QtGui, QtCore


def toolbar_widget():
    toolbar = QtWidgets.QToolBar()
    toolbar.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    toolbar.setStyleSheet('QToolBar {background-color: #121212; color: white;}')
    return toolbar


def dock_widget(name=''):
    dock = QtWidgets.QDockWidget(name)
    dock.setStyleSheet('''
        QDockWidget {
            background-color: #121212;
            font-weight: bold;
            font-size: 12px;
            color: #CCCCCC;
        }
        QDockWidget::title {
            text-align: center;
            background-color: #1E1E1E;
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
            background-color: #1E1E1E;
            border: 0px solid #1E1E1E;
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
                background-color: #121212;
                color: white;
            }
            QFileDialog QLabel {
                color: white;
            }
            QFileDialog QLineEdit {
                background-color: #1E1E1E;
                color: white;
                selection-background-color: #0096FF;
            }
            QFileDialog QPushButton {
                background-color: #1E1E1E;
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
                                    background-color: #1E1E1E;
                                    color: white;
                                }
                            """)
    content_widget = QtWidgets.QWidget(scroll_area)
    content_widget.setStyleSheet("background-color: #1E1E1E;")
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
    label.setStyleSheet('background-color: #1E1E1E; color: #ECECEC; padding: 2px; border-radius: 2px;')
    return label


def lcdnumber_widget(num=None, n=None):
    lcd = QtWidgets.QLCDNumber()
    lcd.setStyleSheet("""
        QLCDNumber {
            background-color: #121212;
            color: white;
            border: 1px solid #333333;
        }
    """)
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
    spinbox.setStyleSheet("""
        QSpinBox {
            background-color: #121212;
            color: white;
            border: 1px solid #333333;
            padding: 2px;
        }
        QSpinBox::up-button {
            background-color: #353535;
            border-left: 1px solid #333333;
        }
        QSpinBox::down-button {
            background-color: #353535;
            border-left: 1px solid #333333;
        }
        QSpinBox::up-arrow {
            image: url(up_arrow.png);  /* You can set a custom arrow image */
        }
        QSpinBox::down-arrow {
            image: url(down_arrow.png);  /* You can set a custom arrow image */
        }
        QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off {
            image: url(up_arrow_disabled.png);  /* Custom arrow image when disabled */
        }
        QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off {
            image: url(down_arrow_disabled.png);  /* Custom arrow image when disabled */
        }
    """)
    spinbox.setRange(range_min, range_max)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    spinbox.setMinimumWidth(spinbox.sizeHint().width())
    spinbox.setMinimumHeight(spinbox.sizeHint().height())
    max_value_width = spinbox.fontMetrics().horizontalAdvance(str(spinbox.maximum()))
    max_value_height = spinbox.fontMetrics().height()
    spinbox.setMaximumWidth(max_value_width + 32)
    spinbox.setMaximumHeight(max_value_height + 16)
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    doublespinbox.setStyleSheet("""
        QDoubleSpinBox {
            background-color: #121212;
            color: white;
            border: 1px solid #333333;
            padding: 2px;
        }
        QToolTip {
            color: white;
            background-color: #2a2a2a;
            border: 1px solid white;
        }
        QDoubleSpinBox::up-button {
            background-color: #353535;
            border-left: 1px solid #333333;
        }
        QDoubleSpinBox::down-button {
            background-color: #353535;
            border-left: 1px solid #333333;
        }
        QDoubleSpinBox::up-arrow {
            image: url(up_arrow.png);  /* You can set a custom arrow image */
        }
        QDoubleSpinBox::down-arrow {
            image: url(down_arrow.png);  /* You can set a custom arrow image */
        }
        QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {
            image: url(up_arrow_disabled.png);  /* Custom arrow image when disabled */
        }
        QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {
            image: url(down_arrow_disabled.png);  /* Custom arrow image when disabled */
        }
    """)
    doublespinbox.setRange(range_min, range_max)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    doublespinbox.setValue(value)
    doublespinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    doublespinbox.setMinimumWidth(doublespinbox.sizeHint().width())
    doublespinbox.setMinimumHeight(doublespinbox.sizeHint().height())
    max_value_width = doublespinbox.fontMetrics().horizontalAdvance(str(doublespinbox.maximum()))
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
            background-color: #121212;
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
            background-color: #121212;
            color: #FFFFFF;
            padding: 2px;
        }
        QCheckBox::indicator {
            width: 25px;
            height: 25px;
            border-radius: 4px;
            border: 2px solid #AAAAAA;
            background-color: #121212;
        }
        QCheckBox::indicator:checked {
            background-color: #4169e1;
            border: 2px solid #4169e1;
        }
    ''')
    checkbox.setChecked(False)
    checkbox.setCheckable(True)
    return checkbox


def radiobutton_widget(name='', color=f"rgb(192, 255, 62)", autoex=False):
    radiobutton = QtWidgets.QRadioButton(name)
    radiobutton.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    radiobutton.setAutoExclusive(autoex)
    radiobutton.setStyleSheet(f'''
        QRadioButton {{
            background-color: #1E1E1E;
            color: white;
        }}
        QRadioButton::indicator {{
            width: 8px;
            height: 8px;
        }}
        QRadioButton::indicator::unchecked {{
            border: 2px solid rgb(200, 200, 200);
            border-radius: 4px;
        }}
        QRadioButton::indicator::checked {{
            background-color: {color};
            border: 2px solid {color};
            border-radius: 4px;
        }}
    ''')
    return radiobutton


def combobox_widget(list_items):
    combobox = QtWidgets.QComboBox()
    for item in list_items:
        combobox.addItem(item)
    combobox.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    combobox.setStyleSheet('''
        QComboBox {
            background-color: #121212;
            color: #FFFFFF;
            border: 1px solid #555555;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
        }
        QComboBox QAbstractItemView {
            background-color: #121212;
            color: #FFFFFF;
            selection-background-color: #4169e1;
        }
    ''')
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
            background-color: #121212;
            color: #FFFFFF;
        }
        QLabel {
            color: #FFFFFF;
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
            background-color: #121212;
            color: #EEEEEE;
            text-align: center;
        }
        QLabel {
            color: #EEEEEE;
        }
    """)
    return msg
