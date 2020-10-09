#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/18/20 6:04 PM
# @Author  : joshon
# @Site    : hsae
# @File    : xml_test.py
# @Software: PyCharm
#use the DOM4j to parse the

import xml.etree.ElementTree as ET

tree=ET.parse("../data/voc2012_raw/VOCdevkit/VOC2012/Annotations/2007_000032.xml")
root=tree.getroot()
print('root-tag:',root.tag,',root-attrib:',root.attrib,',root-text:',root.text)
results={}
def parsexml1(child):
    for child in root:
        print('child-tag是：', child.tag, ',child.attrib：', child.attrib, ',child.text：', child.text)
        if child.tag=="folder":
            results[child.tag]=child.text
        if child.tag=="filename":
            results[child.tag]=child.text
        if child.tag=="source":
            temp_reuslt={}
            for sub_child in child:
                temp_reuslt[sub_child.tag]=sub_child.text
            results[child.tag]=temp_reuslt
        if child.tag == "size":
            temp_reuslt={}
            for sub_child in child:
                temp_reuslt[sub_child.tag]=sub_child.text
            results[child.tag]=temp_reuslt
        if child.tag=="segmented":
            results[child.tag]=child.text
        if child.tag=="object":
            temp_reuslt = {}
            if child.tag not in results:
                results[child.tag]=[]
            for sub_child in child:
                if not len(sub_child):
                    temp_reuslt[sub_child.tag]=sub_child.text
                else:
                    temp_reuslt1={}
                    for sub_sub_child in sub_child:
                        temp_reuslt1[sub_sub_child.tag]=sub_sub_child.text
                    temp_reuslt[sub_child.tag]=temp_reuslt1
            results[child.tag].append(temp_reuslt)
    print(results)




def parsexml(xml):
    if not  len(xml):
        return {xml.tag: xml.text}
    result={}
    for child in xml:
        tem_result=parsexml(child)
        if child.tag != "object":
            result[child.tag]=tem_result[child.tag]
        else:
            if 'object' not  in result:
                result[child.tag]=[]
            result[child.tag].append(tem_result[child.tag])
    return {xml.tag : result}
aa=parsexml(root)
print(aa)
























