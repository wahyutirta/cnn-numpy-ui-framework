# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 19:35:37 2021

@author: ASUS
"""

<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>888</width>
    <height>470</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MainWindow</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_5">
    <item row="0" column="0">
     <layout class="QGridLayout" name="gridLayout_4">
      <item row="1" column="1">
       <widget class="QGroupBox" name="groupBox">
        <property name="title">
         <string>Parameters</string>
        </property>
        <layout class="QGridLayout" name="gridLayout_3">
         <item row="0" column="0">
          <layout class="QGridLayout" name="gridLayout_2">
           <item row="0" column="0">
            <widget class="QLabel" name="label_2">
             <property name="text">
              <string>Audio Device</string>
             </property>
            </widget>
           </item>
           <item row="0" column="1">
            <widget class="QComboBox" name="comboBox"/>
           </item>
           <item row="1" column="0">
            <widget class="QLabel" name="label_3">
             <property name="text">
              <string>Window Length</string>
             </property>
            </widget>
           </item>
           <item row="1" column="1">
            <widget class="QLineEdit" name="lineEdit">
             <property name="text">
              <string>1000</string>
             </property>
            </widget>
           </item>
           <item row="2" column="0">
            <widget class="QLabel" name="label_4">
             <property name="text">
              <string>Sampling Rate (Hz)</string>
             </property>
            </widget>
           </item>
           <item row="2" column="1">
            <widget class="QLineEdit" name="lineEdit_2">
             <property name="text">
              <string>44100</string>
             </property>
            </widget>
           </item>
          </layout>
         </item>
         <item row="0" column="1">
          <layout class="QVBoxLayout" name="verticalLayout">
           <item>
            <layout class="QGridLayout" name="gridLayout">
             <item row="0" column="0">
              <widget class="QLabel" name="label_5">
               <property name="text">
                <string>Down Sample</string>
               </property>
              </widget>
             </item>
             <item row="0" column="1">
              <widget class="QLineEdit" name="lineEdit_3">
               <property name="text">
                <string>1</string>
               </property>
              </widget>
             </item>
             <item row="1" column="0">
              <widget class="QLabel" name="label_6">
               <property name="text">
                <string>Update Interval (ms)</string>
               </property>
              </widget>
             </item>
             <item row="1" column="1">
              <widget class="QLineEdit" name="lineEdit_4">
               <property name="text">
                <string>30</string>
               </property>
              </widget>
             </item>
            </layout>
           </item>
           <item>
            <widget class="QPushButton" name="pushButton">
             <property name="text">
              <string>Plot It!</string>
             </property>
            </widget>
           </item>
          </layout>
         </item>
        </layout>
       </widget>
      </item>
      <item row="2" column="1">
       <widget class="QWidget" name="widget" native="true">
        <property name="styleSheet">
         <string notr="true">background-color: rgb(0, 0, 0);</string>
        </property>
       </widget>
      </item>
      <item row="0" column="1">
       <widget class="QLabel" name="label">
        <property name="text">
         <string>&lt;html&gt;&lt;head/&gt;&lt;body&gt;&lt;p&gt;&lt;span style=&quot; font-size:14pt; font-weight:600;&quot;&gt;PyShine Live Voice Plot GUI&lt;/span&gt;&lt;/p&gt;&lt;/body&gt;&lt;/html&gt;</string>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
       </widget>
      </item>
      <item row="3" column="1">
       <spacer name="horizontalSpacer">
        <property name="orientation">
         <enum>Qt::Horizontal</enum>
        </property>
        <property name="sizeHint" stdset="0">
         <size>
          <width>778</width>
          <height>0</height>
         </size>
        </property>
       </spacer>
      </item>
      <item row="2" column="0">
       <spacer name="verticalSpacer">
        <property name="orientation">
         <enum>Qt::Vertical</enum>
        </property>
        <property name="sizeHint" stdset="0">
         <size>
          <width>0</width>
          <height>40</height>
         </size>
        </property>
       </spacer>
      </item>
     </layout>
    </item>
   </layout>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
</ui>