#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
 @Script     : 
 @IsAvailable: 
 @Time       : 2020/12/29 19:25
 @Author     : 剑怜情
'''
import hashlib
string='123'.encode('utf-8')
print(hashlib.sha1(string))
print(hashlib.sha1(string).hexdigest())