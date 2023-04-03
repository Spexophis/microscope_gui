from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
from utilities import customized_widgets as cw

app = QApplication([])

label1 = cw.label_widget(str('Label 1'))
spinbox1 = cw.spinbox_widget(0, 100, 1, 0)
button1 = cw.pushbutton_widget('Button 1')
label2 = cw.label_widget('Label 2')
spinbox2 = cw.spinbox_widget(0, 100, 1, 0)
button2 = cw.pushbutton_widget('Button 2')

layout = QVBoxLayout()
hbox1 = QHBoxLayout()
hbox1.addWidget(label1)
hbox1.addWidget(spinbox1)
hbox1.addWidget(button1)
layout.addLayout(hbox1)

hbox2 = QHBoxLayout()
hbox2.addWidget(label2)
hbox2.addWidget(spinbox2)
hbox2.addWidget(button2)
layout.addLayout(hbox2)

widget = QWidget()
widget.setLayout(layout)
widget.show()

app.exec_()