#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 01 2023
@author: cpdong
"""

from pymol import cmd

set_color conf1, [0, 83, 214] #50
set_color conf2, [101, 203, 243] #50-70
set_color conf3, [255, 219, 19] #70-90
set_color conf4, [255, 125, 69] #90

def plddt(selection="all"):
    cmd.color("conf4", f"({selection}) and b > 90")
    cmd.color("conf3", f"({selection}) and b < 90 and b > 70")
    cmd.color("conf2", f"({selection}) and b < 70 and b > 50")
    cmd.color("conf1", f"({selection}) and b < 50")
    
cmd.extend("plddt", plddt)
cmd.auto_arg[0]["plddt"] = [cmd.object_sc, "object", ""]
#
