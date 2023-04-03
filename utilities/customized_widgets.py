from PyQt5 import QtWidgets


def group_widget(name=''):
    group = QtWidgets.QGroupBox(name)
    group.setStyleSheet("font: bold Arial 12px")
    return group


def label_widget(name=''):
    label = QtWidgets.QLabel(name)
    label.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    return label


def lcdnumber_widget(n=None):
    lcd = QtWidgets.QLCDNumber()
    lcd.setStyleSheet("background-color: dark; color: white; font: bold Arial 12px")
    lcd.setDecMode()
    lcd.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
    if n is not None:
        lcd.setDigitCount(n)
    lcd.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    lcd.setMinimumWidth(lcd.sizeHint().width())
    return lcd


def spinbox_widget(range_min, range_max, step, value):
    spinbox = QtWidgets.QSpinBox()
    spinbox.setStyleSheet("background-color: dark; color: black; font: bold Arial 12px")
    spinbox.setRange(range_min, range_max)
    spinbox.setSingleStep(step)
    spinbox.setValue(value)
    spinbox.setMaximum(range_max)
    spinbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    spinbox.setMinimumWidth(spinbox.sizeHint().width())
    return spinbox


def doublespinbox_widget(range_min, range_max, step, decimals, value):
    doublespinbox = QtWidgets.QDoubleSpinBox()
    doublespinbox.setStyleSheet("background-color: dark; color: black; font: bold Arial 12px")
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
    button.setStyleSheet("background-color: grey; color: white; font: bold Arial 12px")
    button_size = button.fontMetrics().boundingRect(button.text()).size()
    button.setFixedSize(button_size.width() + 24, button_size.height() + 12)
    if checkable:
        button.setCheckable(True)
        button.setStyleSheet('QPushButton:checked { background-color: #66CDAA; }')
    return button


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
