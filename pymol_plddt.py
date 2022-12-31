#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 01 2023
@author: cpdong
"""

from pymol import cmd

cmd.set_color("conf4", [0, 83, 214]) #90
cmd.set_color("conf3", [101, 203, 243]) #70-90
cmd.set_color("conf2", [255, 219, 19]) #50-70
cmd.set_color("conf1", [255, 125, 69]) #50

def plddt(selection="all"):
    cmd.bg_color("white")
    cmd.color("conf4", f"({selection}) and b > 90")
    cmd.color("conf3", f"({selection}) and b < 90 and b > 70")
    cmd.color("conf2", f"({selection}) and b < 70 and b > 50")
    cmd.color("conf1", f"({selection}) and b < 50")
    
cmd.extend("plddt", plddt)
cmd.auto_arg[0]["plddt"] = [cmd.object_sc, "object", ""]
#
