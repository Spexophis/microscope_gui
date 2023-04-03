from PyQt5 import QtWidgets


def group_widget(name=''):
    group = QtWidgets.QGroupBox(name)
    group.setStyleSheet("""
    QGroupBox {font-weight: bold;}
    QGroupBox:title { color: rgb(240, 255, 255);}""")
    return group


def label_widget(name=''):
    label = QtWidgets.QLabel(name)
    label.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return label


def lcdnumber_widget(num=None, n=None):
    lcd = QtWidgets.QLCDNumber()
    lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    lcd.setStyleSheet("QLCDNumber { background-color: black; color: white; border: 2px solid grey; font: bold Arial 12px; }")
    lcd.setDecMode()
    if num is not None:
        lcd.display(num)
    if n is not None:
        lcd.setDigitCount(n)
    lcd.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    lcd.setMinimumWidth(lcd.sizeHint().width())
    return lcd


def spinbox_widget(range_min, range_max, step, value):
    spinbox = QtWidgets.QSpinBox()
    spinbox.setStyleSheet('''
        QSpinBox { 
        background-color: white; 
        color: black; 
        font: bold Arial 12px;
        border: 1px solid grey; 
        } 

        QSpinBox::up-button, 
        QSpinBox::down-button { 
        width: 0px; 
        }"
    ''')
    spinbox.setRange(range_min, range_max)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setMaximum(range_max)
    spinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    spinbox.setMinimumWidth(spinbox.sizeHint().width())
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setStyleSheet('''
        QDoubleSpinBox { 
        background-color: white; 
        color: black; 
        font: bold Arial 12px;
        border: 1px solid grey; 
        } 
        
        QDoubleSpinBox::up-button, 
        QDoubleSpinBox::down-button { 
        width: 0px; 
        }"
    ''')
    doublespinbox.setRange(-range_min, range_max)
    doublespinbox.setSingleStep(step)
    doublespinbox.setDecimals(decimals)
    doublespinbox.setValue(value)
    doublespinbox.setMaximum(range_max)
    doublespinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    doublespinbox.setMinimumWidth(doublespinbox.sizeHint().width())
    return doublespinbox


def pushbutton_widget(name='', checkable=False):
    button = QtWidgets.QPushButton(name)
    button.setCheckable(checkable)
    button_size = button.fontMetrics().boundingRect(button.text()).size()
    button.setFixedSize(button_size.width() + 24, button_size.height() + 16)
    button.setStyleSheet('''
        QPushButton {
            background-color: #104E8B;
            border-style: outset;
            border-radius: 6px;
            font: bold 12px;
            color: white;
        }
        
        QPushButton:hover {
            background-color: #3e8e41;
        }
        
        QPushButton:pressed {
            background-color: #2e6d3b;
            border-style: inset;
        }
        
        QPushButton:checked {
            background-color: #2e6d3b;
            border-style: inset;
        }
    ''')
    return button


def checkbox_widget(name=''):
    checkbox = QtWidgets.QCheckBox(name)
    button_style = '''
        QCheckBox {
            background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5, 
                stop: 0.8 #fff, stop: 0.9 #ddd);
            color: #000;
            font: 14px;
            padding: 5px;
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
    '''
    checkbox.setStyleSheet(button_style)
    checkbox.setChecked(False)
    checkbox.setCheckable(True)
    return checkbox

def radiobutton_widget(name=''):
    button = QtWidgets.QRadioButton(name)
    button.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return button


def combobox_widget(list_items):
    combobox = QtWidgets.QComboBox()
    for item in list_items:
        combobox.addItem(item)
    combobox.setStyleSheet("background-color: lightgrey; color: black; font: bold Arial 12px")
    return combobox


def lineedit_widget():
    lineedit = QtWidgets.QLineEdit()
    lineedit.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return lineedit


def checkbox_widget(name=''):
    checkbox = QtWidgets.QCheckBox(name)
    checkbox.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return checkbox


def toolbar_widget():
    toolbar = QtWidgets.QToolBar()
    toolbar.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px; width: 36px; height: 18px")
    return toolbar
